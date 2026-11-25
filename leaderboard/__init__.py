"""多源单调聚合精确 Top-K 分析查询模块。

核心设计见 README.md：
- sources.py  版本化数据源、不可变快照、一对多归约
- scoring.py  单调组合分数、缺项规则、稳定并列约定
- engine.py   TA / NRA 执行器，维护未见项上界与停止证据
- baseline.py 全量关联基线与朴素局部 TopK（反例对照）
- datasets.py 内置反例数据集与随机生成器
"""

from .sources import Row, VersionedSource, Snapshot, VersionDrift
from .scoring import ScoreSpec, AGGREGATORS, NEUTRALS
from .engine import TopKEngine, QueryConfig, execute, Ledger
from .baseline import full_join, naive_local_top_union
from . import datasets

__all__ = [
    "Row",
    "VersionedSource",
    "Snapshot",
    "VersionDrift",
    "ScoreSpec",
    "AGGREGATORS",
    "NEUTRALS",
    "TopKEngine",
    "QueryConfig",
    "execute",
    "Ledger",
    "full_join",
    "naive_local_top_union",
    "datasets",
]
