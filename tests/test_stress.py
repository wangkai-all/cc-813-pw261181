"""高强度随机属性测试：大量随机配置下执行器必须与全量关联逐名一致。

运行: python -I tests/test_stress.py
"""

import sys
import time

sys.path.insert(0, "/workspace")

from leaderboard.baseline import compare_rankings, full_join, full_scan_reads
from leaderboard.datasets import random_dataset
from leaderboard.engine import QueryConfig, execute
from leaderboard.scoring import AGGREGATORS


def main() -> int:
    n_cases = 0
    early_stops = 0
    sa_total_engine = 0
    sa_total_full = 0
    t0 = time.time()
    for seed in range(120):
        for agg in AGGREGATORS:
            for missing in ("neutral", "exclude"):
                d = random_dataset(
                    seed=seed,
                    n_sources=2 + seed % 3,
                    n_keys=15 + seed % 40,
                    sparsity=0.1 + (seed % 5) * 0.12,
                    duplicate_p=0.1 + (seed % 4) * 0.1,
                    aggregator=agg,
                    missing=missing,
                )
                k = 1 + seed % 12
                for mode in ("ta", "nra"):
                    cfg = QueryConfig(k=k, mode=mode)
                    result = execute(d.sources, d.spec, cfg)
                    baseline = full_join(d.sources, d.spec, k)
                    report = compare_rankings(result.ranking, baseline)
                    if not report["exact_match"]:
                        print("FAIL", seed, agg, missing, mode, k)
                        print(report)
                        return 1
                    n_cases += 1
                    fc = full_scan_reads(d.sources)
                    sa_total_engine += result.ledger.total_sa
                    sa_total_full += fc["total_sa"]
                    if result.ledger.total_sa < fc["total_sa"]:
                        early_stops += 1
    dt = time.time() - t0
    print(f"通过 {n_cases} 个随机配置，用时 {dt:.1f}s")
    print(f"其中 {early_stops} 个案例实现了提前停止")
    print(
        f"SA 总读取: 执行器 {sa_total_engine} vs 全量 {sa_total_full} "
        f"（节省 {100 * (1 - sa_total_engine / sa_total_full):.1f}%）"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
