"""标准库实现的分析查询 HTTP API（零第三方依赖）。

端点：
GET  /api/datasets                内置数据集清单与当前配置
POST /api/load                    载入数据集（可覆盖权重/缺项/聚合器/模式/K）
POST /api/run                     一次性执行 TA/NRA，返回排名、停止证据、成本
POST /api/step/start              创建步进会话
POST /api/step/next               推进一步，返回当前候选、阈值、未见上界
POST /api/baseline                全量关联基线 + 朴素局部 TopK 对照 + 逐名核对
POST /api/mutate                  模拟查询中源版本替换（continue/restart 验证）
GET  /                            静态网页
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from . import datasets as ds_mod
from .baseline import (
    compare_rankings,
    full_join,
    full_join_all_scores,
    full_scan_reads,
    naive_local_top_union,
)
from .engine import QueryConfig, TopKEngine
from .scoring import AGGREGATORS, ScoreSpec
from .sources import Row, VersionedSource

_STATIC_DIR = "static"


class StepSession:
    def __init__(self, engine: TopKEngine, dataset_name: str):
        self.engine = engine
        self.dataset_name = dataset_name
        self.lock = threading.Lock()


class APIState:
    def __init__(self) -> None:
        self.dataset: Optional[ds_mod.Dataset] = None
        self.dataset_name: str = ""
        self.sessions: Dict[str, StepSession] = {}
        self._seq = 0
        self.lock = threading.Lock()

    def new_session_id(self) -> str:
        self._seq += 1
        return f"sess-{self._seq}"


STATE = APIState()


# ---------------------------------------------------------------------------
# 请求处理辅助
# ---------------------------------------------------------------------------


def _json_response(handler: BaseHTTPRequestHandler, payload: Any, status: int = 200) -> None:
    body = json.dumps(payload, ensure_ascii=False, default=_json_default).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, frozenset):
        return sorted(obj)
    if isinstance(obj, set):
        return sorted(obj)
    raise TypeError(f"不可序列化: {type(obj)}")


def _read_json(handler: BaseHTTPRequestHandler) -> dict:
    length = int(handler.headers.get("Content-Length", 0))
    if not length:
        return {}
    raw = handler.rfile.read(length)
    return json.loads(raw.decode("utf-8"))


def _spec_from_request(dataset: ds_mod.Dataset, req: dict) -> ScoreSpec:
    base = dataset.spec
    weights = req.get("weights") or dict(base.weights)
    return ScoreSpec(
        aggregator=req.get("aggregator", base.aggregator),
        weights={k: float(v) for k, v in weights.items()},
        missing=req.get("missing", base.missing),
        domain_max=dict(base.domain_max) if base.domain_max else {},
    )


def _config_from_request(req: dict) -> QueryConfig:
    return QueryConfig(
        k=int(req.get("k", 10)),
        mode=req.get("mode", "ta"),
        sa_policy=req.get("sa_policy", "round_robin"),
        on_drift=req.get("on_drift", "continue"),
        max_rounds=req.get("max_rounds"),
    )


def _latencies_from_request(req: dict, sources: List[VersionedSource]) -> Dict[str, float]:
    raw = req.get("latencies") or {}
    return {s.name: float(raw.get(s.name, 0.0)) for s in sources}


def _serialize_sources(dataset: ds_mod.Dataset) -> List[dict]:
    out = []
    for s in dataset.sources:
        snap = s.snapshot()
        out.append(
            {
                "name": s.name,
                "version": snap.version,
                "reducer": s.reducer,
                "n_rows": len(snap.rows),
                "n_keys": len(snap.sorted_scan()),
                "cost_sa": s.cost_sa,
                "cost_ra": s.cost_ra,
                "top_entries": [
                    {"key": e.key, "score": e.score, "n_rows": e.n_rows}
                    for e in snap.sorted_scan()[:5]
                ],
            }
        )
    return out


# ---------------------------------------------------------------------------
# 业务处理
# ---------------------------------------------------------------------------


def handle_datasets() -> dict:
    items = []
    for name, factory in ds_mod.BUILTIN.items():
        d = factory()
        items.append(
            {
                "name": name,
                "description": d.description,
                "k": d.k,
                "spec": {
                    "aggregator": d.spec.aggregator,
                    "weights": dict(d.spec.weights),
                    "missing": d.spec.missing,
                },
                "sources": _serialize_sources(d),
                "expected_winners": list(d.expected_winners),
                "naive_misses": list(d.naive_misses),
                "notes": d.notes,
            }
        )
    current = None
    if STATE.dataset is not None:
        current = {
            "name": STATE.dataset_name,
            "spec": {
                "aggregator": STATE.dataset.spec.aggregator,
                "weights": dict(STATE.dataset.spec.weights),
                "missing": STATE.dataset.spec.missing,
            },
            "k": STATE.dataset.k,
            "sources": _serialize_sources(STATE.dataset),
        }
    return {"datasets": items, "aggregators": list(AGGREGATORS), "current": current}


def handle_load(req: dict) -> dict:
    name = req.get("name", "counterexample")
    with STATE.lock:
        STATE.dataset = ds_mod.load(name)
        STATE.dataset_name = name
        STATE.sessions.clear()
    if "aggregator" in req or "weights" in req or "missing" in req:
        STATE.dataset.spec = _spec_from_request(STATE.dataset, req)
    if "k" in req:
        STATE.dataset.k = int(req["k"])
    return {
        "loaded": name,
        "description": STATE.dataset.description,
        "spec": {
            "aggregator": STATE.dataset.spec.aggregator,
            "weights": dict(STATE.dataset.spec.weights),
            "missing": STATE.dataset.spec.missing,
        },
        "k": STATE.dataset.k,
        "sources": _serialize_sources(STATE.dataset),
    }


def _require_dataset() -> ds_mod.Dataset:
    if STATE.dataset is None:
        STATE.dataset = ds_mod.load("counterexample")
        STATE.dataset_name = "counterexample"
    return STATE.dataset


def _select_dataset(req: dict) -> ds_mod.Dataset:
    """按请求体的 name 选择数据集；未指定时用当前已载入的数据集。

    注意不能忽略 name，否则针对 A 数据集的请求会误用全局当前数据集 B。
    每次按 name 现取一份独立副本，保证 mutate 实验不污染内置数据集与其他请求。
    """
    name = req.get("name")
    if name and name != STATE.dataset_name:
        return ds_mod.load(name)
    return _require_dataset()


def handle_run(req: dict) -> dict:
    dataset = _select_dataset(req)
    spec = _spec_from_request(dataset, req)
    config = _config_from_request(req)
    lat = _latencies_from_request(req, dataset.sources)
    engine = TopKEngine(dataset.sources, spec, config, latencies=lat)
    result = engine.run()

    baseline = full_join(dataset.sources, spec, config.k)
    check = compare_rankings(result.ranking, baseline)
    full_cost = full_scan_reads(dataset.sources)
    return {
        "result": result.as_dict(),
        "baseline_topk": [{"key": k, "score": s} for k, s in baseline],
        "correctness": check,
        "full_scan_cost": full_cost,
        "savings": _savings(result.ledger, full_cost),
    }


def _savings(ledger, full_cost: dict) -> dict:
    sa_saved = full_cost["total_sa"] - ledger.total_sa
    return {
        "sa_saved": sa_saved,
        "sa_saved_pct": round(100 * sa_saved / max(full_cost["total_sa"], 1), 2),
        "engine_virtual_time": round(ledger.virtual_time, 4),
        "full_virtual_time": full_cost["virtual_time"],
        "time_saved_pct": round(
            100
            * (full_cost["virtual_time"] - ledger.virtual_time)
            / max(full_cost["virtual_time"], 1e-9),
            2,
        ),
    }


def handle_step_start(req: dict) -> dict:
    dataset = _require_dataset()
    spec = _spec_from_request(dataset, req)
    config = _config_from_request(req)
    lat = _latencies_from_request(req, dataset.sources)
    engine = TopKEngine(dataset.sources, spec, config, latencies=lat)
    sid = STATE.new_session_id()
    STATE.sessions[sid] = StepSession(engine, STATE.dataset_name)
    return {
        "session_id": sid,
        "mode": config.mode,
        "k": config.k,
        "sources": [s.name for s in dataset.sources],
        "spec": {
            "aggregator": spec.aggregator,
            "weights": dict(spec.weights),
            "missing": spec.missing,
        },
    }


def handle_step_next(req: dict) -> dict:
    sid = req["session_id"]
    session = STATE.sessions.get(sid)
    if session is None:
        raise KeyError(f"步进会话 {sid!r} 不存在（可能已因重新载入被清除）")
    with session.lock:
        engine = session.engine
        event = engine.step()
        if event is None:
            return {"finished": True}
        return {
            "finished": engine.finished,
            "event": event,
            "current_topk": [
                {"key": k, "score": s} for k, s in engine.current_topk()
            ],
            "unseen_ub": event.get("unseen_ub"),
            "ledger": engine.ledger.as_dict(),
            "stop_evidence": engine.stop_evidence,
        }


def handle_baseline(req: dict) -> dict:
    dataset = _select_dataset(req)
    spec = _spec_from_request(dataset, req)
    k = int(req.get("k", dataset.k))
    full = full_join(dataset.sources, spec, k)
    full_all = full_join_all_scores(dataset.sources, spec)
    naive = naive_local_top_union(dataset.sources, spec, k)
    naive_keys = {k_ for k_, _ in naive}
    missed = [
        {"rank": i + 1, "key": key, "score": score}
        for i, (key, score) in enumerate(full)
        if key not in naive_keys
    ]
    return {
        "full_topk": [{"key": k_, "score": s} for k_, s in full],
        "full_ranking_all": [
            {"key": k_, "score": s} for k_, s in full_all
        ],
        "naive_local_topk": [{"key": k_, "score": s} for k_, s in naive],
        "naive_missed_winners": missed,
        "n_objects": len(full_all),
    }


def handle_mutate(req: dict) -> dict:
    """模拟查询进行中的源版本替换。

    body: {"source": "sales", "rows": [[key, score], ...] | null}
    rows=null 时用扰动后的当前数据替换。返回新旧版本号，供 continue/restart 验证。
    """
    dataset = _require_dataset()
    source_name = req["source"]
    target = next((s for s in dataset.sources if s.name == source_name), None)
    if target is None:
        raise KeyError(f"数据源 {source_name!r} 不存在")
    old_version = target.current_version
    old_snap = target.snapshot()

    rows_req = req.get("rows")
    if rows_req is None:
        # 默认扰动：对每个分数做确定性小幅修改，制造版本差异
        import random as _r

        rng = _r.Random(old_version * 1000 + 7)
        new_rows = [
            Row(
                key=r.key,
                score=round(max(0.0, min(1.0, r.score + rng.uniform(-0.1, 0.1))), 4),
                row_id=r.row_id,
                payload=r.payload,
            )
            for r in old_snap.rows
        ]
    else:
        new_rows = [
            Row(key=str(k), score=float(v), row_id=i) for i, (k, v) in enumerate(rows_req)
        ]
    new_version = target.replace(new_rows)

    affected = []
    for sid, sess in STATE.sessions.items():
        if sess.dataset_name != STATE.dataset_name:
            continue
        held = sess.engine.snaps.get(source_name)
        if held is not None and held.version == old_version:
            affected.append(
                {
                    "session_id": sid,
                    "held_snapshot_version": old_version,
                    "policy": sess.engine.config.on_drift,
                }
            )
    return {
        "source": source_name,
        "old_version": old_version,
        "new_version": new_version,
        "active_sessions": affected,
        "note": "进行中的会话仍持有旧快照；下一步将按 on_drift=continue/restart 处理",
    }


# ---------------------------------------------------------------------------
# HTTP Handler
# ---------------------------------------------------------------------------


class _Handler(BaseHTTPRequestHandler):
    server_version = "LeaderboardTA/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:  # 安静日志
        return

    def do_GET(self) -> None:  # noqa: N802 - stdlib 命名
        path = urlparse(self.path).path
        try:
            if path == "/api/datasets":
                _json_response(self, handle_datasets())
                return
            if path in ("/", "/index.html"):
                self._serve_static("index.html", "text/html; charset=utf-8")
                return
            if path == "/app.js":
                self._serve_static("app.js", "application/javascript; charset=utf-8")
                return
            if path == "/styles.css":
                self._serve_static("styles.css", "text/css; charset=utf-8")
                return
            _json_response(self, {"error": f"未知路径 {path}"}, 404)
        except Exception as exc:  # noqa: BLE001 - API 层统一兜底
            _json_response(self, {"error": str(exc), "type": type(exc).__name__}, 500)

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        routes = {
            "/api/load": handle_load,
            "/api/run": handle_run,
            "/api/step/start": handle_step_start,
            "/api/step/next": handle_step_next,
            "/api/baseline": handle_baseline,
            "/api/mutate": handle_mutate,
        }
        handler = routes.get(path)
        if handler is None:
            _json_response(self, {"error": f"未知路径 {path}"}, 404)
            return
        try:
            req = _read_json(self)
            _json_response(self, handler(req))
        except KeyError as exc:
            _json_response(self, {"error": str(exc), "type": "KeyError"}, 404)
        except ValueError as exc:
            _json_response(self, {"error": str(exc), "type": "ValueError"}, 400)
        except Exception as exc:  # noqa: BLE001
            _json_response(self, {"error": str(exc), "type": type(exc).__name__}, 500)

    def _serve_static(self, filename: str, content_type: str) -> None:
        import os

        path = os.path.join(os.path.dirname(__file__), _STATIC_DIR, filename)
        with open(path, "rb") as fh:
            body = fh.read()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def build_server(host: str = "127.0.0.1", port: int = 8000) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), _Handler)


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    httpd = build_server(host, port)
    print(f"排行榜分析查询服务已启动: http://{host}:{port}")
    httpd.serve_forever()
