"""精确 Top-K 执行器：TA 与 NRA 两种访问策略。

TA（threshold algorithm，``mode="ta"``）
    每轮对一个源做一次排序访问（SA），随后对该业务键在其余未知源上做随机访问（RA）
    立即补全；维护所有 *未见* 对象（任何源都还没读到的键）可达的上界
    ``UB_unseen``，只有当当前第 K 名的确定分数严格高于该上界时才提前停止。

NRA（no random access，``mode="nra"``）
    只做排序访问，为每个已见键维护 (下界, 上界) 区间；当临时前 K 名全部确定、
    且其余任何对象（含未见对象）的上界都严格低于第 K 名分数时停止。

两种模式都保证：停止时给出的前 K 项与全量关联（:func:`leaderboard.baseline.full_join`）
完全一致；停止证据记录在 ``ledger`` 与返回结果中。

版本一致性
    引擎首次访问时为每个源固定 :class:`~leaderboard.sources.Snapshot`。
    计算期间源被替换：``on_drift="continue"`` 继续读旧快照（记录漂移事件）；
    ``on_drift="restart"`` 整次重开（状态与账本清零，restart 计数 +1）。
    任何情况下都不会混合两个版本的数据。
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from .scoring import EPS, ScoreSpec, compare_rank
from .sources import ReducedEntry, Snapshot, VersionedSource


# ---------------------------------------------------------------------------
# 成本账本：记录真实读取量
# ---------------------------------------------------------------------------


@dataclass
class Ledger:
    """按源记录真实读取量与虚拟耗时。"""

    sa_count: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    ra_count: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    raw_rows_touched: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    virtual_time: float = 0.0
    restarts: int = 0
    trace: List[dict] = field(default_factory=list)

    def charge_sa(self, source: str, cost: float, n_rows: int, latency: float) -> None:
        self.sa_count[source] += 1
        self.raw_rows_touched[source] += n_rows
        self.virtual_time += cost + latency

    def charge_ra(self, source: str, cost: float, n_rows: int, latency: float) -> None:
        self.ra_count[source] += 1
        self.raw_rows_touched[source] += n_rows
        self.virtual_time += cost + latency

    @property
    def total_sa(self) -> int:
        return sum(self.sa_count.values())

    @property
    def total_ra(self) -> int:
        return sum(self.ra_count.values())

    def as_dict(self) -> dict:
        return {
            "sa_count": dict(self.sa_count),
            "ra_count": dict(self.ra_count),
            "raw_rows_touched": dict(self.raw_rows_touched),
            "total_sa": self.total_sa,
            "total_ra": self.total_ra,
            "virtual_time": round(self.virtual_time, 4),
            "restarts": self.restarts,
            "events": len(self.trace),
        }


# ---------------------------------------------------------------------------
# 查询配置与结果
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QueryConfig:
    k: int = 10
    mode: str = "ta"                 # ta | nra
    sa_policy: str = "round_robin"   # round_robin | cheapest
    on_drift: str = "continue"       # continue | restart
    max_rounds: Optional[int] = None

    def __post_init__(self) -> None:
        if self.mode not in ("ta", "nra"):
            raise ValueError(f"未知模式: {self.mode!r}")
        if self.sa_policy not in ("round_robin", "cheapest"):
            raise ValueError(f"未知 SA 调度策略: {self.sa_policy!r}")
        if self.on_drift not in ("continue", "restart"):
            raise ValueError(f"未知漂移策略: {self.on_drift!r}")
        if self.k <= 0:
            raise ValueError("k 必须为正整数")


@dataclass
class QueryResult:
    ranking: List[Tuple[str, float]]
    ledger: Ledger
    rounds: int
    stop_evidence: dict
    restarts: int
    mode: str

    def as_dict(self) -> dict:
        return {
            "ranking": [{"key": k, "score": s} for k, s in self.ranking],
            "ledger": self.ledger.as_dict(),
            "rounds": self.rounds,
            "stop_evidence": self.stop_evidence,
            "restarts": self.restarts,
            "mode": self.mode,
        }


# ---------------------------------------------------------------------------
# 执行器
# ---------------------------------------------------------------------------


class TopKEngine:
    """流式执行器：:meth:`step` 每轮返回一个事件，:meth:`run` 跑到停止。"""

    def __init__(
        self,
        sources: Sequence[VersionedSource],
        spec: ScoreSpec,
        config: QueryConfig,
        *,
        latencies: Optional[Mapping[str, float]] = None,
    ):
        if not sources:
            raise ValueError("至少需要一个数据源")
        self.sources = list(sources)
        self.source_names = [s.name for s in self.sources]
        if len(set(self.source_names)) != len(self.source_names):
            raise ValueError("数据源名称必须唯一")
        self.spec = spec
        self.config = config
        self.latencies: Dict[str, float] = dict(latencies or {})
        self.ledger = Ledger()
        self.round = 0
        self.finished = False
        self.stop_evidence: Optional[dict] = None
        self._init_state()

    # -- 状态初始化（重启时复用） ------------------------------------------
    def _init_state(self) -> None:
        self.snaps: Dict[str, Snapshot] = {s.name: s.snapshot() for s in self.sources}
        # 域上界：用户显式配置优先；否则从快照的归约视图自动探测，
        # 避免 sum 归约/权重使分数超过默认 1.0 而低估未见上界
        detected = {
            name: (max((e.score for e in snap.sorted_scan()), default=0.0))
            for name, snap in self.snaps.items()
        }
        self._domain_max = {
            name: float(self.spec.domain_max.get(name, detected[name]))
            for name in self.source_names
        }
        self.scans: Dict[str, List[ReducedEntry]] = {
            n: snap.sorted_scan() for n, snap in self.snaps.items()
        }
        self.cursor: Dict[str, int] = {n: 0 for n in self.source_names}
        self.threshold: Dict[str, float] = {}
        # 已见键的状态：确认在场的值 / 确认缺席的源
        self.known: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.absent: Dict[str, set] = defaultdict(set)
        self._rr_idx = 0
        self.ledger = Ledger(restarts=self.ledger.restarts)
        # 保留重启前的轨迹摘要
        if self.round:
            self.ledger.trace.append(
                {"type": "restart", "round": self.round, "restarts": self.ledger.restarts}
            )

    # -- 漂移检测 -----------------------------------------------------------
    def _drifted(self) -> List[Tuple[str, int, int]]:
        out = []
        for src in self.sources:
            snap_v = self.snaps[src.name].version
            if src.current_version != snap_v:
                out.append((src.name, snap_v, src.current_version))
        return out

    def _handle_drift(self) -> Optional[dict]:
        drifted = self._drifted()
        if not drifted:
            return None
        if self.config.on_drift == "continue":
            ev = {
                "type": "drift_continue",
                "round": self.round,
                "drifted": [
                    {"source": n, "snapshot_version": ev_, "current_version": cv}
                    for n, ev_, cv in drifted
                ],
                "note": "继续读取查询开始时固定的旧快照，不混合新版本数据",
            }
            self.ledger.trace.append(ev)
            return ev
        # restart：整次重开
        self.ledger.restarts += 1
        ev = {
            "type": "drift_restart",
            "round": self.round,
            "restarts": self.ledger.restarts,
            "drifted": [
                {"source": n, "snapshot_version": ev_, "current_version": cv}
                for n, ev_, cv in drifted
            ],
            "note": "源版本变更，查询整次重开，旧读取全部作废",
        }
        self.round = 0
        self.finished = False
        self.stop_evidence = None
        self._init_state()
        self.ledger.trace.append(ev)
        return ev

    # -- 读取原语 -----------------------------------------------------------
    def _sa(self, name: str) -> Optional[ReducedEntry]:
        snap = self.snaps[name]
        i = self.cursor[name]
        if i >= len(self.scans[name]):
            return None
        entry = self.scans[name][i]
        self.cursor[name] = i + 1
        self.ledger.charge_sa(
            name, snap.cost_sa, entry.n_rows, self.latencies.get(name, 0.0)
        )
        # SA 同时确认了该键在本源的归约分
        self.known[entry.key][name] = entry.score
        self.threshold[name] = entry.score
        return entry

    def _ra(self, name: str, key: str) -> Optional[ReducedEntry]:
        snap = self.snaps[name]
        entry = snap.lookup(key)
        self.ledger.charge_ra(
            name, snap.cost_ra, entry.n_rows if entry else 0,
            self.latencies.get(name, 0.0),
        )
        if entry is None:
            self.absent[key].add(name)
        else:
            self.known[key][name] = entry.score
        return entry

    def _unknown_sources(self, key: str) -> frozenset:
        """该键状态仍未知的源；扫描已耗尽的源惰性记为缺席。"""
        kn = self.known.get(key, {})
        ab = self.absent[key]
        out = []
        for s in self.source_names:
            if s in kn or s in ab:
                continue
            if self.cursor[s] >= len(self.scans[s]):
                ab.add(s)  # 归约视图里没有此键 => 确认缺席
            else:
                out.append(s)
        return frozenset(out)

    def _threshold_caps(self) -> Dict[str, float]:
        """阈值封顶：每个源 ``min(dmax, 当前阈值)``。

        仅在 NRA（纯排序访问）下对 *未知状态* 合法：未知状态只能由该源后续的
        SA 揭示，而 SA 按归约分降序移动，游标之后的键分数不超过当前阈值。

        TA 下不可用：RA 只补全已见键，完全未见的键可能在某些源 *缺席*
        （中性元对 product=1、min=+∞ 不抬分），只在阈值更高的少数源在场取值，
        其聚合分可以超过阈值组合，阈值封顶会低估上界并漏掉真正赢家。
        """
        return {
            s: min(self._domain_max[s], self.threshold.get(s, self._domain_max[s]))
            for s in self.source_names
        }

    def _unseen_caps(self) -> Dict[str, float]:
        """完全未见对象的逐源 cap。

        已发生过 SA 的源（游标 > 0）：未见键若在场必在游标之后，分数 ≤ 阈值，
        故 cap = min(dmax, 阈值)；它也可能缺席，由 :meth:`nra_bounds` 一并枚举。
        尚未开始扫描的源：没有阈值信息，cap = 探测到的域上界 dmax。
        TA 与 NRA 同一规则；区别只在 NRA 所有源都会很快进入"已扫描"状态。
        """
        caps = {}
        for s in self.source_names:
            d = self._domain_max[s]
            if self.cursor[s] > 0 and s in self.threshold:
                caps[s] = min(d, self.threshold[s])
            else:
                caps[s] = d
        return caps

    def _seen_caps(self) -> Dict[str, float]:
        """已见键的未知源上界。

        NRA：未知状态只能由后续 SA 揭示，阈值封顶合法。
        TA：RA 按键直接取值、不经过排序游标，已见键在未知源的实际值可能高于
        当前阈值（该源还没 SA 到它），阈值封顶会低估上界、漏掉赢家，
        因此只能使用固定域上界 dmax。
        """
        if self.config.mode == "ta":
            return dict(self._domain_max)
        return self._threshold_caps()

    def _tight_spec(self, caps: Mapping[str, float]) -> ScoreSpec:
        """用引擎探测后的域上界构造内部 ScoreSpec。"""
        spec = self.spec
        return ScoreSpec(
            aggregator=spec.aggregator,
            weights=spec.weights,
            missing=spec.missing,
            domain_max=dict(caps),
            tie_break=spec.tie_break,
        )

    def _bounds(
        self, key: str, caps: Optional[Mapping[str, float]] = None
    ) -> Tuple[float, float]:
        unknown = self._unknown_sources(key)
        if self.config.mode == "nra":
            # NRA：未知源可能缺席，也可能在场且值不超过 SA 阈值
            return self.spec.nra_bounds(
                self.known.get(key, {}),
                unknown,
                self.threshold,
                domain_override=self._domain_max,
            )
        # TA：RA 直取，未知源只可能取 [0, dmax] 或缺席，阈值不参与
        tightened = self._tight_spec(self._seen_caps())
        return tightened.bounds(self.known.get(key, {}), unknown)

    def _final_score(self, key: str) -> Optional[float]:
        unknown = self._unknown_sources(key)
        if unknown:
            return None
        return self.spec.complete_score(self.known.get(key, {}), self.source_names)

    # -- SA 调度 ------------------------------------------------------------
    def _pick_source(self) -> Optional[str]:
        active = [
            n for n in self.source_names
            if self.cursor[n] < len(self.scans[n])
        ]
        if not active:
            return None
        if self.config.sa_policy == "cheapest":
            return min(active, key=lambda n: (self.snaps[n].cost_sa, n))
        # round robin：从上次位置向后找第一个未耗尽的源
        for off in range(len(self.source_names)):
            n = self.source_names[(self._rr_idx + off) % len(self.source_names)]
            if n in active:
                self._rr_idx = (self._rr_idx + off + 1) % len(self.source_names)
                return n
        return None  # pragma: no cover - active 非空时不可达

    # -- 单轮执行 -----------------------------------------------------------
    def step(self) -> Optional[dict]:
        if self.finished:
            return None
        drift_ev = self._handle_drift()
        self.round += 1
        if self.config.max_rounds and self.round > self.config.max_rounds:
            raise RuntimeError(f"超过 max_rounds={self.config.max_rounds} 仍未收敛")

        name = self._pick_source()
        event: dict = {"round": self.round, "mode": self.config.mode, "drift": drift_ev}
        if name is None:
            # 所有源耗尽：所有候选键均可判定
            event["action"] = "exhausted"
            unseen_caps = self._unseen_caps()
            event["thresholds"] = dict(self.threshold)
            event["unseen_ub"] = self._unseen_ub(unseen_caps)
            event["candidates"] = self._candidate_snapshot()
            self.ledger.trace.append(event)
            self._finish("all_sources_exhausted", event)
            return event

        entry = self._sa(name)
        event["action"] = "sa"
        event["source"] = name
        event["entry"] = {"key": entry.key, "score": entry.score, "n_rows": entry.n_rows}

        if self.config.mode == "ta":
            ra_log = []
            for s in self._unknown_sources(entry.key):
                found = self._ra(s, entry.key)
                ra_log.append(
                    {
                        "source": s,
                        "found": found is not None,
                        "score": found.score if found else None,
                    }
                )
            event["random_access"] = ra_log

        unseen_caps = self._unseen_caps()
        event["thresholds"] = dict(self.threshold)
        event["unseen_ub"] = self._unseen_ub(unseen_caps)
        event["candidates"] = self._candidate_snapshot()
        self.ledger.trace.append(event)

        stop = self._maybe_stop(unseen_caps, event)
        if stop:
            self._finish(stop["rule"], event, stop)
        return event

    def _unseen_ub(self, caps: Mapping[str, float]) -> float:
        """任何源都还没 SA 到的键可达的最高分数。

        逐源 cap：已扫描源为 min(dmax, 阈值)，未扫描源为 dmax；
        每个源仍可能缺席（中性元/exclude），由 nra_bounds 统一枚举两种可能，
        对 wsum/product/min/max 与两种缺项规则都成立。
        """
        _, ub = self.spec.nra_bounds(
            {},
            frozenset(self.source_names),
            self.threshold,
            domain_override=self._domain_max,
            caps_override=dict(caps),
        )
        return ub

    def _candidate_snapshot(self) -> List[dict]:
        out = []
        for key in sorted(self.known):
            lb, ub = self._bounds(key)
            final = self._final_score(key)
            out.append(
                {
                    "key": key,
                    "lb": lb,
                    "ub": ub,
                    "final": final,
                    "known": dict(self.known[key]),
                    "absent": sorted(self.absent[key]),
                    "unknown": sorted(self._unknown_sources(key)),
                }
            )
        return out

    # -- 停止判定 -----------------------------------------------------------
    def _resolved_ranking(self) -> List[Tuple[str, float]]:
        scored = []
        for key in self.known:
            s = self._final_score(key)
            if s is not None:
                scored.append((key, s))
        scored.sort(key=lambda t: _RankKey(t[1], t[0]))
        return scored

    def _maybe_stop(self, caps: Mapping[str, float], event: dict) -> Optional[dict]:
        k = self.config.k
        unseen_ub = self._unseen_ub(caps)

        if self.config.mode == "ta":
            ranking = self._resolved_ranking()
            event["resolved_topk"] = ranking[:k]
            if len(ranking) >= k:
                bkey, bscore = ranking[k - 1]
                # 并列约定为 id 升序：未见对象可能取更小 id，
                # 只有上界严格低于边界分数（超出 EPS）才能保证不漏
                if unseen_ub + EPS < bscore:
                    return {
                        "rule": "threshold_bound",
                        "boundary_key": bkey,
                        "boundary_score": bscore,
                        "unseen_ub": unseen_ub,
                        "thresholds": dict(self.threshold),
                        "text": (
                            f"第 {k} 名 {bkey} 确定分数 {bscore:.6f} 严格高于 "
                            f"任何未见对象的上界 {unseen_ub:.6f}，提前停止"
                        ),
                    }
            return None

        # NRA：临时前 K 名（按下界排序），要求 K 名胜出者全部确定，
        # 且其他候选与未见对象的上界严格低于第 K 名分数
        provisional = []
        for key in self.known:
            lb, ub = self._bounds(key)  # NRA 下内部即阈值封顶
            provisional.append((key, lb, ub))
        provisional.sort(key=lambda t: _RankKey(t[1], t[0]))
        event["provisional_topk"] = [(k_, lb) for k_, lb, _ in provisional[:k]]
        if len(provisional) < k:
            return None
        winners = provisional[:k]
        if any(self._unknown_sources(key) for key, _, _ in winners):
            return None  # 胜出者尚未全部确定，不能给精确结果
        winner_scores = []
        for key, _, _ in winners:
            final = self._final_score(key)
            if final is None:
                return None  # exclude 规则下可能整体不可上榜
            winner_scores.append((key, final))
        bkey, bscore = winner_scores[-1]
        for key, _, ub in provisional[k:]:
            if not _strictly_below(ub, bscore, bkey, key):
                return None
        if not _strictly_below(unseen_ub, bscore, bkey, None):
            return None
        return {
            "rule": "nra_interval_closure",
            "boundary_key": bkey,
            "boundary_score": bscore,
            "unseen_ub": unseen_ub,
            "winners": [{"key": k_, "score": s_} for k_, s_ in winner_scores],
            "thresholds": dict(self.threshold),
            "text": (
                f"临时前 {k} 名全部确定，第 {k} 名 {bkey} 分数 {bscore:.6f}；"
                f"其余对象上界最高 {unseen_ub:.6f}，区间闭包，提前停止"
            ),
        }

    def _finish(self, rule: str, last_event: dict, stop: Optional[dict] = None) -> None:
        self.finished = True
        ranking = self._resolved_ranking()[: self.config.k]
        if stop is None:
            stop = {
                "rule": rule,
                "thresholds": dict(self.threshold),
                "text": "所有源排序访问耗尽，全部候选状态确定，自然结束",
            }
        stop["round"] = self.round
        self.stop_evidence = stop
        last_event["stop"] = stop

    # -- 跑到结束 -----------------------------------------------------------
    def run(self) -> QueryResult:
        while not self.finished:
            self.step()
        ranking = self._resolved_ranking()[: self.config.k]
        return QueryResult(
            ranking=ranking,
            ledger=self.ledger,
            rounds=self.round,
            stop_evidence=self.stop_evidence or {},
            restarts=self.ledger.restarts,
            mode=self.config.mode,
        )

    def current_topk(self) -> List[Tuple[str, float]]:
        return self._resolved_ranking()[: self.config.k]


# ---------------------------------------------------------------------------
# 排序辅助
# ---------------------------------------------------------------------------


class _RankKey:
    """配合稳定并列约定的排序键（EPS 内分数视为并列，键升序）。"""

    __slots__ = ("score", "key")

    def __init__(self, score: float, key: str):
        self.score = score
        self.key = key

    def __lt__(self, other: "_RankKey") -> bool:
        return compare_rank(self.score, self.key, other.score, other.key) < 0

    def __eq__(self, other: object) -> bool:  # pragma: no cover
        return isinstance(other, _RankKey) and self.score == other.score and self.key == other.key


def _strictly_below(ub: float, bscore: float, bkey: str, other_key: Optional[str]) -> bool:
    """对象上界 ub 是否不可能胜过边界 (bscore, bkey)。

    未见对象 other_key=None：id 未知，可能取最小 id，故要求 ub 严格小于分数。
    已见对象：id 已知，可利用稳定并列约定做完整比较。
    """
    if other_key is None:
        return ub + EPS < bscore
    # compare_rank(a, b) < 0 表示 a 排在 b 前；这里要求 (bscore, bkey) 严格在
    # (ub, other_key) 前面，即 other 最好情况也只能排在边界之后。
    return compare_rank(bscore, bkey, ub, other_key) < 0


def execute(
    sources: Sequence[VersionedSource],
    spec: ScoreSpec,
    config: QueryConfig,
    *,
    latencies: Optional[Mapping[str, float]] = None,
) -> QueryResult:
    """一次性执行查询的便捷函数。"""
    return TopKEngine(sources, spec, config, latencies=latencies).run()
