"""版本化数据源、不可变快照与一对多归约。

每张表（源）由若干 :class:`Row` 组成，业务键 ``key`` 标识排行榜对象，
同一业务键在同一源内可出现多次（一对多）。

访问方式（与算法文献一致）：

* 排序访问（sorted access, SA）：按归约后的分数从高到低逐行读取；
* 随机访问（random access, RA）：给定业务键，取出该键在本源的全部原始行。

源数据可以被替换（版本递增）。任何进行中的查询通过首次访问时拿到的
:class:`Snapshot` 持续读取旧版本——快照是不可变视图，因此绝不会出现
"半张新表半张旧表" 的拼版读取。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# 归约函数：一对多 -> 每个 (源, 业务键) 一个标量
# ---------------------------------------------------------------------------

REDUCERS: Dict[str, Callable[[List[float]], float]] = {
    "sum": sum,
    "max": max,
    "avg": lambda xs: sum(xs) / len(xs),
    # 确定性首值：按 (score 降序, row_id 升序) 取第一行，避免随机顺序影响结果
    "first": None,  # 在 Row 层处理，占位以便校验集合完整
}


@dataclass(frozen=True)
class Row:
    """源内一行原始数据。"""

    key: str
    score: float
    row_id: int
    payload: str = ""


@dataclass(frozen=True)
class ReducedEntry:
    """一个业务键在某源内归约后的单条结果（排序访问返回它）。"""

    key: str
    score: float
    n_rows: int


class VersionDrift(RuntimeError):
    """查询期间检测到底层源版本变更（仅 restart 策略会抛出/触发重开）。"""

    def __init__(self, source: str, expected: int, actual: int):
        super().__init__(
            f"源 {source!r} 版本变更: 查询持有 v{expected}, 当前 v{actual}"
        )
        self.source = source
        self.expected = expected
        self.actual = actual


# ---------------------------------------------------------------------------
# 不可变快照
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Snapshot:
    """某源在某一版本下的不可变视图。

    快照持有构造时刻的数据引用；:class:`VersionedSource` 替换数据时整体换对象，
    旧快照继续指向旧数据，天然隔离写入。
    """

    name: str
    version: int
    rows: Tuple[Row, ...]
    reducer: str
    cost_sa: float
    cost_ra: float
    # 归约视图在首次使用时构建并缓存（对象内部惰性，外部语义仍不可变）
    _by_key: Dict[str, Tuple[Row, ...]] = field(init=False, default_factory=dict)
    _sorted: Tuple[ReducedEntry, ...] = field(init=False, default_factory=tuple)

    def __post_init__(self) -> None:
        by_key: Dict[str, List[Row]] = {}
        for row in self.rows:
            by_key.setdefault(row.key, []).append(row)
        frozen = {k: tuple(v) for k, v in by_key.items()}
        reduced: List[ReducedEntry] = []
        for key, rows_k in frozen.items():
            reduced.append(
                ReducedEntry(
                    key=key,
                    score=_reduce(rows_k, self.reducer),
                    n_rows=len(rows_k),
                )
            )
        # 排序访问顺序：归约分降序，键升序保证确定性
        reduced.sort(key=lambda e: (-e.score, e.key))
        object.__setattr__(self, "_by_key", frozen)
        object.__setattr__(self, "_sorted", tuple(reduced))

    # -- 排序访问 -----------------------------------------------------------
    def sorted_scan(self) -> List[ReducedEntry]:
        """整张归约视图（执行器按游标逐次消费）。"""
        return list(self._sorted)

    # -- 随机访问 -----------------------------------------------------------
    def lookup(self, key: str) -> Optional[ReducedEntry]:
        rows_k = self._by_key.get(key)
        if rows_k is None:
            return None
        return ReducedEntry(
            key=key, score=_reduce(rows_k, self.reducer), n_rows=len(rows_k)
        )

    @property
    def keys(self) -> List[str]:
        return list(self._by_key.keys())


def _reduce(rows: List[Row], reducer: str) -> float:
    scores = [r.score for r in rows]
    if reducer == "sum":
        return sum(scores)
    if reducer == "max":
        return max(scores)
    if reducer == "avg":
        return sum(scores) / len(scores)
    if reducer == "first":
        chosen = sorted(rows, key=lambda r: (-r.score, r.row_id))[0]
        return chosen.score
    raise ValueError(f"未知归约规则: {reducer!r}，支持 {sorted(set(REDUCERS) | {'first'})}")


# ---------------------------------------------------------------------------
# 版本化源
# ---------------------------------------------------------------------------


class VersionedSource:
    """带版本号的数据源。

    ``replace`` 整体替换数据并递增版本；进行中的查询仍持有旧 :class:`Snapshot`，
    可通过 :meth:`current_version` 检测漂移并决定 continue / restart。
    """

    def __init__(
        self,
        name: str,
        rows: List[Row],
        *,
        reducer: str = "max",
        cost_sa: float = 1.0,
        cost_ra: float = 5.0,
    ):
        if reducer not in REDUCERS:
            raise ValueError(f"未知归约规则: {reducer!r}")
        self.name = name
        self.reducer = reducer
        self.cost_sa = float(cost_sa)
        self.cost_ra = float(cost_ra)
        self._version = 1
        self._snapshot = self._build(rows)

    def _build(self, rows: List[Row]) -> Snapshot:
        return Snapshot(
            name=self.name,
            version=self._version,
            rows=tuple(rows),
            reducer=self.reducer,
            cost_sa=self.cost_sa,
            cost_ra=self.cost_ra,
        )

    def snapshot(self) -> Snapshot:
        """拿到当前版本的不可变快照。"""
        return self._snapshot

    def replace(self, rows: List[Row]) -> int:
        """整体替换数据，返回新版本号（禁止原地改，避免拼版）。"""
        self._version += 1
        self._snapshot = self._build(rows)
        return self._version

    @property
    def current_version(self) -> int:
        return self._version

    def __repr__(self) -> str:  # pragma: no cover - 调试辅助
        return (
            f"VersionedSource(name={self.name!r}, version={self._version}, "
            f"rows={len(self._snapshot.rows)}, reducer={self.reducer!r})"
        )
