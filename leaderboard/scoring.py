"""单调组合分数、缺项规则与稳定并列约定。

支持的聚合器（对各源归约后的标量 x_s 组合，均为单调非降函数）：

``wsum``    加权和（缺项 exclude 时按在场源权重重新归一化）
``min``     最小值
``max``     最大值
``product`` 乘积

缺项规则（:class:`ScoreSpec.missing`）：

``neutral`` 缺项取该聚合器的中性元（wsum=0、min=+∞、max=-∞、product=1），
            即 "没有贡献"，但源本身仍占位；
``exclude`` 缺项直接移出组合，按剩余源计算（wsum 重新归一化权重）。

所有源分数假定非负，且每个源有域上界 ``domain_max``（默认 1.0），
未见源的上界据此计算。

并列约定（稳定）：总分相同（浮点容差 EPS 内视为并列）时按业务键字典序升序，
结果完全确定，与访问顺序无关。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Mapping, Optional, Sequence, Tuple

EPS = 1e-12

# 各聚合器在 neutral 缺项规则下的中性元
NEUTRALS: Dict[str, float] = {
    "wsum": 0.0,
    "min": math.inf,
    "max": -math.inf,
    "product": 1.0,
}

AGGREGATORS = tuple(NEUTRALS.keys())


@dataclass(frozen=True)
class ScoreSpec:
    """一次查询的分数定义。"""

    aggregator: str = "wsum"
    weights: Mapping[str, float] = field(default_factory=dict)
    missing: str = "neutral"          # neutral | exclude
    domain_max: Mapping[str, float] = field(default_factory=dict)
    tie_break: str = "id_asc"         # 目前仅支持业务键升序的稳定约定

    def __post_init__(self) -> None:
        if self.aggregator not in NEUTRALS:
            raise ValueError(f"未知聚合器: {self.aggregator!r}，支持 {AGGREGATORS}")
        if self.missing not in ("neutral", "exclude"):
            raise ValueError(f"未知缺项规则: {self.missing!r}")
        if self.tie_break != "id_asc":
            raise ValueError("仅支持 id_asc 稳定并列约定")

    # -- 配置查询 -----------------------------------------------------------
    def weight(self, source: str) -> float:
        return float(self.weights.get(source, 1.0))

    def dmax(self, source: str) -> float:
        return float(self.domain_max.get(source, 1.0))

    # -- 完整分数（每个源的在场/缺席都已知） --------------------------------
    def complete_score(
        self,
        present: Mapping[str, float],
        all_sources: Sequence[str],
    ) -> Optional[float]:
        """计算对象的最终分数。

        ``present`` 为确认在场的 源->归约分；其余源视为确认缺席。
        exclude 规则下所有源都缺席时返回 ``None``（对象不可上榜）。
        """
        if self.missing == "exclude":
            vals = {s: v for s, v in present.items()}
            if not vals:
                return None
            return self._aggregate(vals, renorm_wsum=True)
        filled: Dict[str, float] = {}
        neutral = NEUTRALS[self.aggregator]
        for s in all_sources:
            filled[s] = present[s] if s in present else neutral
        return self._aggregate(filled, renorm_wsum=False)

    def _aggregate(self, vals: Mapping[str, float], *, renorm_wsum: bool) -> float:
        agg = self.aggregator
        if agg == "wsum":
            if renorm_wsum:
                wsum = sum(self.weight(s) * v for s, v in vals.items())
                wtotal = sum(self.weight(s) for s in vals)
                return wsum / wtotal if wtotal > 0 else 0.0
            return sum(self.weight(s) * v for s, v in vals.items())
        if agg == "min":
            return min(vals.values())
        if agg == "max":
            return max(vals.values())
        # product
        p = 1.0
        for v in vals.values():
            p *= v
        return p

    # -- 界：known 为已确认在场的 源->分，unknown 为状态未知的源 -----------
    def bounds(
        self,
        known: Mapping[str, float],
        unknown: FrozenSet[str],
    ) -> Tuple[float, float]:
        """返回 (下界, 上界)。未知源可能取 [0, domain_max] 内任意值或缺席。"""
        if not unknown:
            score = self._aggregate(known, renorm_wsum=self.missing == "exclude")
            return score, score
        if self.missing == "neutral":
            return self._bounds_neutral(known, unknown)
        return self._bounds_exclude(known, unknown)

    def nra_bounds(
        self,
        known: Mapping[str, float],
        unknown: FrozenSet[str],
        thresholds: Mapping[str, float],
        domain_override: Optional[Mapping[str, float]] = None,
        caps_override: Optional[Mapping[str, float]] = None,
    ) -> Tuple[float, float]:
        """阈值感知界：未知源要么缺席，要么在场且分数不超过该源的 cap。

        未知源若在场，其值必在 ``[0, cap]`` 内；但它同样可能 *缺席*
        （取中性元或被 exclude），两种情况都纳入枚举。

        - NRA：cap = min(dmax, 当前 SA 阈值)，由 ``thresholds`` 给出；
        - TA 未见对象：已扫描源 cap=阈值（键在游标之后），未扫描源 cap=dmax，
          由 ``caps_override`` 逐源指定；
        - TA 已见对象：走 :meth:`bounds`，cap 统一为 dmax。
        """
        if not unknown:
            score = self._aggregate(known, renorm_wsum=self.missing == "exclude")
            return score, score
        dmax = domain_override or {}

        def _dmax(s: str) -> float:
            return float(dmax.get(s, self.dmax(s)))

        if caps_override is not None:
            t = {s: float(caps_override[s]) for s in unknown}
        else:
            t = {
                s: min(_dmax(s), thresholds.get(s, _dmax(s)))
                for s in unknown
            }
        pvals = list(known.values())
        agg = self.aggregator

        if self.missing == "neutral":
            if agg == "wsum":
                base = sum(self.weight(s) * v for s, v in known.items())
                return base, base + sum(self.weight(s) * t[s] for s in unknown)
            if agg == "min":
                if not pvals:
                    return 0.0, math.inf        # 全部缺席 => +inf；任一在场取 0 => 0
                return 0.0, min(pvals)
            if agg == "max":
                if not pvals:
                    return -math.inf, max(t.values())
                return max(pvals), max(max(pvals), max(t.values()))
            # product / neutral：缺席中性元 1，不改变乘积。
            # 上界必须覆盖"未知源全部缺席" => 乘积保持 base，
            # 也覆盖全部在场取阈值 => base*Πt，取两者最大。
            base = math.prod(pvals) if pvals else 1.0
            upper = max(base, base * math.prod(t.values()))
            return 0.0, upper

        # ---- exclude：缺席源直接移出组合 ----
        if agg == "wsum":
            wp = sum(self.weight(s) for s in known)
            num = sum(self.weight(s) * v for s, v in known.items())
            # 下界：未知源全部在场取 0，稀释均值（无在场源则对象可能不可上榜）
            lower = (
                num / (wp + sum(self.weight(s) for s in unknown))
                if wp > 0 else -math.inf
            )
            # 上界：按阈值降序贪心纳入能抬高均值的未知源
            best = num / wp if wp > 0 else -math.inf
            cur_num, cur_w = num, wp
            for s in sorted(unknown, key=lambda s: -t[s]):
                w = self.weight(s)
                cand = (cur_num + w * t[s]) / (cur_w + w)
                if cand >= best:
                    best = cand
                    cur_num += w * t[s]
                    cur_w += w
            return lower, best
        if agg == "min":
            if not pvals:
                return -math.inf, math.inf
            return 0.0, min(pvals)
        if agg == "max":
            if not pvals:
                return -math.inf, max(t.values())
            return max(pvals), max(max(pvals), max(t.values()))
        # product / exclude：缺席源移出，乘积保持；在场取阈值则乘阈值
        if not pvals:
            return 0.0, max(1.0, math.prod(t.values()))
        base = math.prod(pvals)
        return 0.0, max(base, base * math.prod(t.values()))

    # ------------------------------------------------------------------ #
    def _bounds_neutral(
        self, known: Mapping[str, float], unknown: FrozenSet[str]
    ) -> Tuple[float, float]:
        agg = self.aggregator
        pvals = list(known.values())
        if agg == "wsum":
            base = sum(self.weight(s) * v for s, v in known.items())
            # 未知源缺席 -> 中性元 0；在场最好 -> w * dmax
            upper = base + sum(self.weight(s) * self.dmax(s) for s in unknown)
            return base, upper
        if agg == "min":
            lower = 0.0 if unknown else (min(pvals) if pvals else math.inf)
            upper = min(pvals) if pvals else math.inf
            return lower, upper
        if agg == "max":
            lower = max(pvals) if pvals else -math.inf
            upper = max([max(pvals, default=-math.inf)] + [self.dmax(s) for s in unknown])
            return lower, upper
        # product：中性元 1
        base = 1.0
        for v in pvals:
            base *= v
        lower = 0.0  # 任一未知源在场取 0 即可使乘积为 0
        upper = base
        for s in unknown:
            upper *= self.dmax(s)
        return lower, upper

    def _bounds_exclude(
        self, known: Mapping[str, float], unknown: FrozenSet[str]
    ) -> Tuple[float, float]:
        agg = self.aggregator
        pvals = list(known.values())
        if agg == "wsum":
            wp = sum(self.weight(s) for s in known)
            num = sum(self.weight(s) * v for s, v in known.items())
            wu = sum(self.weight(s) for s in unknown)
            # 上界：按 dmax 降序贪心纳入能抬高均值的未知源
            best = num / wp if wp > 0 else -math.inf
            cur_num, cur_w = num, wp
            for s in sorted(unknown, key=lambda s: -self.dmax(s)):
                d, w = self.dmax(s), self.weight(s)
                cand = (cur_num + w * d) / (cur_w + w)
                if cand >= best:
                    best = cand
                    cur_num, cur_w = cur_num + w * d, cur_w + w
            # 下界：未知源全部在场且取 0，稀释最严重（无在场源时对象可能整体不可上榜）
            lower = num / (wp + wu) if wp > 0 else -math.inf
            return lower, best
        if agg == "min":
            if not pvals:
                return -math.inf, math.inf
            return 0.0, min(pvals)       # 未知源在场取 0 / 全部缺席最有利
        if agg == "max":
            if not pvals:
                return -math.inf, max(self.dmax(s) for s in unknown)
            return max(pvals), max(max(pvals), max(self.dmax(s) for s in unknown))
        # product（exclude）
        if not pvals:
            return 0.0, math.prod(self.dmax(s) for s in unknown)
        base = math.prod(pvals)
        return 0.0, base * math.prod(self.dmax(s) for s in unknown)


# ---------------------------------------------------------------------------
# 稳定并列约定
# ---------------------------------------------------------------------------


def compare_rank(a_score: float, a_key: str, b_score: float, b_key: str) -> int:
    """排行榜排序比较：分数降序；EPS 内视为并列，业务键升序。

    返回 -1 表示 a 排在 b 前面。完全确定性，与访问顺序无关。
    """
    if a_score > b_score + EPS:
        return -1
    if b_score > a_score + EPS:
        return 1
    if a_key == b_key:
        return 0
    return -1 if a_key < b_key else 1
