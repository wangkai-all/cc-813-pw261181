"""内置反例数据集与确定性随机生成器。

``counterexample`` 是需求指定的自带反例：三张源表，每个源的局部前三名都被
"偏科" 对象占据，真正的总赢家 ``star`` 在每个源都只排第 4，朴素局部拼接
必然漏掉它，而单调聚合执行器必须精确找到它。

其余数据集分别覆盖：一对多归约、缺项规则（neutral / exclude）、稳定并列。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .scoring import ScoreSpec
from .sources import Row, VersionedSource


@dataclass
class Dataset:
    name: str
    description: str
    sources: List[VersionedSource]
    spec: ScoreSpec
    k: int
    expected_winners: Tuple[str, ...]
    naive_misses: Tuple[str, ...] = ()
    notes: Dict[str, str] = field(default_factory=dict)

    def clone(self) -> "Dataset":
        """深拷贝源对象（网页端做版本替换实验时不污染原数据集）。"""
        new_sources = []
        for s in self.sources:
            snap = s.snapshot()
            new_sources.append(
                VersionedSource(
                    s.name,
                    list(snap.rows),
                    reducer=s.reducer,
                    cost_sa=s.cost_sa,
                    cost_ra=s.cost_ra,
                )
            )
        return Dataset(
            self.name,
            self.description,
            new_sources,
            self.spec,
            self.k,
            self.expected_winners,
            self.naive_misses,
            dict(self.notes),
        )


def _src(
    name: str,
    rows: Sequence[Tuple[str, float]],
    *,
    reducer: str = "max",
    cost_sa: float = 1.0,
    cost_ra: float = 5.0,
) -> VersionedSource:
    return VersionedSource(
        name,
        [Row(key=k, score=v, row_id=i) for i, (k, v) in enumerate(rows)],
        reducer=reducer,
        cost_sa=cost_sa,
        cost_ra=cost_ra,
    )


# ---------------------------------------------------------------------------
# 1. 核心反例：局部前列漏掉真正赢家
# ---------------------------------------------------------------------------


def counterexample() -> Dataset:
    # 三源等权 wsum，K=3。业务键跨源稀疏（真实多表场景）：
    # - 偏科键 a*/b*/g* 只在各自源出现，霸占每源局部 Top3，但总分只有单源贡献；
    # - star 与 mid1 在三个源都在场且均衡，star 每源排第 4（0.75），
    #   总分 2.25 高居总榜第 1，任何单源局部 Top3 都不含它；
    # - 朴素局部 TopK 拼接既漏掉 star 也漏掉 mid1。
    # 每源 20 个键；未见对象的保守上界可在排序访问耗尽前降到第 K 名分数之下。
    def _sparse_rows(prefix: str) -> List[Tuple[str, float]]:
        rows = [
            (f"{prefix}1", 1.0), (f"{prefix}2", 0.98), (f"{prefix}3", 0.96),
            (f"{prefix}m1", 0.70), (f"{prefix}m2", 0.66), (f"{prefix}m3", 0.62),
            (f"{prefix}m4", 0.55), (f"{prefix}m5", 0.50), (f"{prefix}m6", 0.45),
            (f"{prefix}m7", 0.40), (f"{prefix}m8", 0.35), (f"{prefix}m9", 0.30),
        ]
        # 长尾：把阈值继续拉低，让执行器在扫到这些键之前就能证明停止
        for i in range(1, 13):
            rows.append((f"{prefix}tail{i}", round(0.28 - 0.02 * i, 4)))
        rows.append(("star", 0.75))
        rows.append(("mid1", 0.60))
        return rows

    s1 = _src("sales", _sparse_rows("a"), cost_sa=1.0, cost_ra=4.0)
    s2 = _src("reviews", _sparse_rows("b"), cost_sa=1.5, cost_ra=3.0)
    s3 = _src("traffic", _sparse_rows("g"), cost_sa=2.0, cost_ra=6.0)
    spec = ScoreSpec(
        aggregator="wsum",
        weights={"sales": 1.0, "reviews": 1.0, "traffic": 1.0},
        missing="neutral",
    )
    return Dataset(
        name="counterexample",
        description=(
            "核心反例：star 在每个源都只排第 4，任何单源局部 Top3 都不含它，"
            "但等权加权和 2.25 高居总榜第 1；mid1(1.80) 同样被局部拼接漏掉。"
            "偏科键只在单源出现，每源 20 键，执行器可凭未见上界提前停止。"
        ),
        sources=[s1, s2, s3],
        spec=spec,
        k=3,
        expected_winners=("star", "mid1", "a1"),
        naive_misses=("star", "mid1"),
        notes={
            "star_total": "0.75*3 = 2.25（每源第 4，总榜第 1）",
            "mid1_total": "0.60*3 = 1.80",
            "a1_total": "1.0 + 0 + 0 = 1.0（单源偏科，缺项中性元 0）",
        },
    )


# ---------------------------------------------------------------------------
# 2. 一对多：归约防膨胀
# ---------------------------------------------------------------------------


def one_to_many() -> Dataset:
    # p 在 sales 有 3 行；max 归约后与单行 0.8 等价，sum 归约则虚高。
    s1 = _src(
        "orders",
        [("p", 0.8), ("p", 0.7), ("p", 0.6), ("q", 0.75), ("r", 0.7)],
        reducer="max",
    )
    s2 = _src(
        "ratings",
        [("q", 0.9), ("r", 0.85), ("p", 0.5)],
        reducer="max",
    )
    spec = ScoreSpec(
        aggregator="wsum",
        weights={"orders": 1.0, "ratings": 1.0},
        missing="neutral",
    )
    return Dataset(
        name="one_to_many",
        description=(
            "一对多归约：p 在 orders 有 3 行。max 归约时 p=0.8（不会因关联膨胀霸榜），"
            "可切换 sum 观察虚高；first 归约按 (分数降序,row_id) 稳定取首行。"
        ),
        sources=[s1, s2],
        spec=spec,
        k=3,
        expected_winners=("q", "r", "p"),
        notes={"p_rows": "orders 中 3 行，归约后只占一个榜位"},
    )


# ---------------------------------------------------------------------------
# 3. 缺项：neutral 与 exclude 两种规则给出不同榜单
# ---------------------------------------------------------------------------


def missing_keys() -> Dataset:
    s1 = _src("s1", [("x", 0.9), ("y", 0.6), ("z", 0.6)])
    s2 = _src("s2", [("x", 0.9), ("y", 0.6)])           # z 缺席
    spec_neutral = ScoreSpec(
        aggregator="wsum", weights={"s1": 1.0, "s2": 1.0}, missing="neutral"
    )
    return Dataset(
        name="missing_keys",
        description=(
            "缺项规则：z 只在 s1 出现。neutral 规则 z=0.6（缺席源贡献 0），"
            "exclude 规则 z=0.6/1=0.6 归一化后与 y 持平并按 id 决胜。"
        ),
        sources=[s1, s2],
        spec=spec_neutral,
        k=3,
        expected_winners=("x", "y", "z"),
        notes={
            "neutral_scores": "x=1.8, y=1.2, z=0.6",
            "exclude_scores": "x=0.9, y=0.6, z=0.6（y/z 并列，id 升序 y 在前）",
        },
    )


# ---------------------------------------------------------------------------
# 4. 稳定并列
# ---------------------------------------------------------------------------


def ties() -> Dataset:
    # alpha 与 beta 加权和完全相等；稳定约定要求 alpha 在前（id 升序）。
    s1 = _src("t1", [("alpha", 0.8), ("beta", 0.6), ("gamma", 0.95)])
    s2 = _src("t2", [("beta", 0.9), ("alpha", 0.7), ("gamma", 0.2)])
    spec = ScoreSpec(
        aggregator="wsum", weights={"t1": 1.0, "t2": 1.0}, missing="neutral"
    )
    return Dataset(
        name="ties",
        description=(
            "稳定并列：alpha(0.8+0.7) 与 beta(0.6+0.9) 总分都是 1.5，"
            "按业务键字典序 alpha 必须稳定在前，与访问顺序无关。"
        ),
        sources=[s1, s2],
        spec=spec,
        k=3,
        expected_winners=("alpha", "beta", "gamma"),
        notes={"alpha": "1.5", "beta": "1.5", "gamma": "1.15"},
    )


# ---------------------------------------------------------------------------
# 5. 确定性随机数据集（属性测试用）
# ---------------------------------------------------------------------------


def random_dataset(
    seed: int = 42,
    n_sources: int = 3,
    n_keys: int = 40,
    *,
    sparsity: float = 0.25,
    duplicate_p: float = 0.15,
    aggregator: str = "wsum",
    missing: str = "neutral",
) -> Dataset:
    rng = random.Random(seed)
    names = [f"src{i}" for i in range(n_sources)]
    keys = [f"k{i:03d}" for i in range(n_keys)]
    sources: List[VersionedSource] = []
    weights = {n: round(rng.uniform(0.5, 2.0), 3) for n in names}
    for si, name in enumerate(names):
        rows: List[Tuple[str, float]] = []
        rid = 0
        for key in keys:
            if rng.random() < sparsity:
                continue  # 缺项
            rows.append((key, round(rng.uniform(0.0, 1.0), 4)))
            if rng.random() < duplicate_p:  # 一对多
                rows.append((key, round(rng.uniform(0.0, 1.0), 4)))
        sources.append(
            VersionedSource(
                name,
                [Row(k, v, i) for i, (k, v) in enumerate(rows)],
                reducer=rng.choice(["max", "sum", "avg"]),
                cost_sa=round(rng.uniform(0.5, 3.0), 2),
                cost_ra=round(rng.uniform(2.0, 8.0), 2),
            )
        )
    spec = ScoreSpec(aggregator=aggregator, weights=weights, missing=missing)
    return Dataset(
        name=f"random_seed{seed}",
        description=(
            f"确定性随机数据集（seed={seed}）：{n_sources} 源 / {n_keys} 键 / "
            f"缺项率 {sparsity} / 一对多概率 {duplicate_p}"
        ),
        sources=sources,
        spec=spec,
        k=10,
        expected_winners=(),  # 由基线动态核对
    )


BUILTIN = {
    "counterexample": counterexample,
    "one_to_many": one_to_many,
    "missing_keys": missing_keys,
    "ties": ties,
}


def load(name: str) -> Dataset:
    if name not in BUILTIN:
        raise KeyError(f"未知数据集 {name!r}，内置: {sorted(BUILTIN)}")
    return BUILTIN[name]()


def all_builtin() -> List[Dataset]:
    return [f() for f in BUILTIN.values()]
