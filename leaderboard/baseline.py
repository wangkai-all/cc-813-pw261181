"""全量关联基线与朴素局部 TopK（反例对照）。

:func:`full_join` 是 "全算" 基准：对每个源做完整归约，再对全体业务键的
笛卡尔关联结果统一打分排序。它无视访问代价，只保证答案正确，供执行器
逐步核对。

:func:`naive_local_top_union` 实现需求里点名的错误做法——每源各自取前 K，
再把局部榜单简单拼接去重后打分。:mod:`leaderboard.datasets` 中的反例数据集
保证它漏掉真正的赢家。
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Mapping, Sequence, Tuple

from .scoring import EPS, ScoreSpec, compare_rank
from .sources import ReducedEntry, VersionedSource


def _reduced_map(source: VersionedSource) -> Dict[str, ReducedEntry]:
    snap = source.snapshot()
    return {e.key: e for e in snap.sorted_scan()}


def full_join(
    sources: Sequence[VersionedSource],
    spec: ScoreSpec,
    k: int,
) -> List[Tuple[str, float]]:
    """全量关联：读取所有源的全部数据，计算每个对象的精确总分并排序。"""
    per_source: List[Dict[str, ReducedEntry]] = [_reduced_map(s) for s in sources]
    all_keys = set()
    for m in per_source:
        all_keys.update(m)

    scored: List[Tuple[str, float]] = []
    names = [s.name for s in sources]
    for key in all_keys:
        present = {
            names[i]: m[key].score
            for i, m in enumerate(per_source)
            if key in m
        }
        score = spec.complete_score(present, names)
        if score is None:
            continue  # exclude 规则且所有源缺席（理论上不会出现，防御性处理）
        scored.append((key, score))

    scored.sort(key=lambda t: _ExactOrStable(t[1], t[0]))
    return scored[:k]


def full_join_all_scores(
    sources: Sequence[VersionedSource],
    spec: ScoreSpec,
) -> List[Tuple[str, float]]:
    """全量关联的完整排名（不止前 K），供逐名核对。"""
    per_source = [_reduced_map(s) for s in sources]
    all_keys = set()
    for m in per_source:
        all_keys.update(m)
    names = [s.name for s in sources]
    scored = []
    for key in all_keys:
        present = {names[i]: m[key].score for i, m in enumerate(per_source) if key in m}
        score = spec.complete_score(present, names)
        if score is not None:
            scored.append((key, score))
    scored.sort(key=lambda t: _ExactOrStable(t[1], t[0]))
    return scored


def naive_local_top_union(
    sources: Sequence[VersionedSource],
    spec: ScoreSpec,
    k: int,
) -> List[Tuple[str, float]]:
    """错误做法：各源先取局部前 K，再把键的并集拼起来打分。

    反例数据集下它会漏掉任何单源都进不了前 K、但组合总分很高的对象。
    """
    names = [s.name for s in sources]
    local: List[set] = []
    per_source: List[Dict[str, ReducedEntry]] = []
    for source in sources:
        m = _reduced_map(source)
        per_source.append(m)
        local_top = sorted(m.values(), key=lambda e: (-e.score, e.key))[:k]
        local.append({e.key for e in local_top})

    union_keys: set = set().union(*local) if local else set()
    scored = []
    for key in union_keys:
        present = {names[i]: m[key].score for i, m in enumerate(per_source) if key in m}
        score = spec.complete_score(present, names)
        if score is not None:
            scored.append((key, score))
    scored.sort(key=lambda t: _ExactOrStable(t[1], t[0]))
    return scored[:k]


class _ExactOrStable:
    """与执行器一致的稳定排序键（EPS 容差并列 + id 升序）。"""

    __slots__ = ("score", "key")

    def __init__(self, score: float, key: str):
        self.score = score
        self.key = key

    def __lt__(self, other: "_ExactOrStable") -> bool:
        return compare_rank(self.score, self.key, other.score, other.key) < 0

    def __eq__(self, other: object) -> bool:  # pragma: no cover
        return isinstance(other, _ExactOrStable) and self.key == other.key


def compare_rankings(
    engine_topk: Sequence[Tuple[str, float]],
    baseline_topk: Sequence[Tuple[str, float]],
) -> dict:
    """逐名核对执行器结果与全量基线，输出差异报告。"""
    mismatches = []
    for i in range(max(len(engine_topk), len(baseline_topk))):
        e = engine_topk[i] if i < len(engine_topk) else None
        b = baseline_topk[i] if i < len(baseline_topk) else None
        if e is None or b is None or e[0] != b[0] or abs(e[1] - b[1]) > EPS:
            mismatches.append({"rank": i + 1, "engine": e, "baseline": b})
    return {
        "exact_match": not mismatches,
        "engine_n": len(engine_topk),
        "baseline_n": len(baseline_topk),
        "mismatches": mismatches,
    }


def full_scan_reads(sources: Sequence[VersionedSource]) -> dict:
    """全量关联需要的读取量（每个源整张归约视图各一次 SA，无 RA）。"""
    sa = {}
    raw = {}
    for source in sources:
        snap = source.snapshot()
        entries = snap.sorted_scan()
        sa[source.name] = len(entries)
        raw[source.name] = len(snap.rows)
    return {
        "sa_count": sa,
        "ra_count": {s.name: 0 for s in sources},
        "raw_rows_touched": raw,
        "total_sa": sum(sa.values()),
        "total_ra": 0,
        "virtual_time": round(
            sum(sa[s.name] * s.cost_sa for s in sources), 4
        ),
    }
