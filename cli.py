"""命令行演示：所有数字均来自引擎对当前快照的实际执行。

用法:
  python -I cli.py demo                 # 内置数据集完整演示
  python -I cli.py run counterexample --mode nra --k 3
  python -I cli.py stress --seeds 60    # 随机正确性压测
  python -I cli.py drift                # 版本漂移 continue/restart 演示
"""

from __future__ import annotations

import argparse
import sys

sys.path.insert(0, "/workspace")

from leaderboard.baseline import (
    compare_rankings,
    full_join,
    full_scan_reads,
    naive_local_top_union,
)
from leaderboard.datasets import BUILTIN, load, random_dataset
from leaderboard.engine import QueryConfig, TopKEngine, execute
from leaderboard.scoring import ScoreSpec
from leaderboard.sources import Row


def _hr(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def _print_ledger(ledger, full_cost) -> None:
    print(f"{'源':<10}{'SA(引擎/全量)':>16}{'RA':>6}{'触及原始行':>12}")
    for name in ledger.sa_count:
        print(
            f"{name:<10}"
            f"{str(ledger.sa_count[name]) + '/' + str(full_cost['sa_count'][name]):>16}"
            f"{ledger.ra_count[name]:>6}"
            f"{ledger.raw_rows_touched[name]:>12}"
        )
    print(
        f"{'合计':<10}"
        f"{str(ledger.total_sa) + '/' + str(full_cost['total_sa']):>16}"
        f"{ledger.total_ra:>6}"
        f"{sum(ledger.raw_rows_touched.values()):>12}"
    )
    saved = full_cost["total_sa"] - ledger.total_sa
    pct = 100 * saved / max(full_cost["total_sa"], 1)
    print(
        f"\n虚拟耗时: 引擎 {ledger.virtual_time:.2f} vs 全量 {full_cost['virtual_time']:.2f} "
        f"| SA 节省 {saved} 次 ({pct:.1f}%) | 重启 {ledger.restarts} 次"
    )


def cmd_demo(args) -> None:
    for name, factory in BUILTIN.items():
        d = factory()
        _hr(f"数据集 {name}: {d.description}")
        full_cost = full_scan_reads(d.sources)
        baseline = full_join(d.sources, d.spec, d.k)
        naive = naive_local_top_union(d.sources, d.spec, d.k)
        naive_keys = {k for k, _ in naive}
        print(f"全量关联前 {d.k} 名: {[(k, round(s, 4)) for k, s in baseline]}")
        print(f"朴素局部拼接前 {d.k} 名: {[k for k, _ in naive]}")
        missed = [k for k, _ in baseline if k not in naive_keys]
        print(f"朴素拼接漏掉的真正赢家: {missed if missed else '无'}")

        for mode in ("ta", "nra"):
            result = execute(d.sources, d.spec, QueryConfig(k=d.k, mode=mode))
            check = compare_rankings(result.ranking, baseline)
            print(f"\n--- {mode.upper()} 模式 ---")
            print(f"榜单: {[(k, round(s, 4)) for k, s in result.ranking]}")
            print(f"与全量基线逐名核对: {'精确一致 ✓' if check['exact_match'] else check}")
            se = result.stop_evidence
            print(f"停止规则: {se.get('rule')}（第 {result.rounds} 轮）")
            print(f"停止证据: {se.get('text', '')}")
            if "unseen_ub" in se:
                print(f"  未见对象上界 UB={se['unseen_ub']:.6f}, "
                      f"边界分数={se.get('boundary_score'):.6f}")
            _print_ledger(result.ledger, full_cost)


def cmd_run(args) -> None:
    d = load(args.dataset)
    spec = ScoreSpec(
        aggregator=args.aggregator or d.spec.aggregator,
        weights=dict(d.spec.weights),
        missing=args.missing or d.spec.missing,
    )
    cfg = QueryConfig(
        k=args.k, mode=args.mode, sa_policy=args.sa_policy, on_drift=args.on_drift
    )
    lat = {s.name: args.latency for s in d.sources} if args.latency else None
    result = execute(d.sources, spec, cfg, latencies=lat)
    baseline = full_join(d.sources, spec, args.k)
    _hr(f"run {args.dataset} mode={args.mode} k={args.k}")
    for i, (key, score) in enumerate(result.ranking, 1):
        print(f"{i:>3}. {key:<12} {score:.6f}")
    print(f"\n核对: {compare_rankings(result.ranking, baseline)}")
    print(f"停止: {result.stop_evidence.get('rule')} — {result.stop_evidence.get('text','')}")
    _print_ledger(result.ledger, full_scan_reads(d.sources))


def cmd_stress(args) -> None:
    n = ok = early = 0
    sa_eng = sa_full = 0
    for seed in range(args.seeds):
        for agg in ("wsum", "min", "max", "product"):
            for missing in ("neutral", "exclude"):
                d = random_dataset(
                    seed=seed,
                    n_sources=2 + seed % 3,
                    n_keys=15 + seed % 40,
                    sparsity=0.1 + (seed % 5) * 0.12,
                    aggregator=agg,
                    missing=missing,
                )
                k = 1 + seed % 12
                for mode in ("ta", "nra"):
                    r = execute(d.sources, d.spec, QueryConfig(k=k, mode=mode))
                    b = full_join(d.sources, d.spec, k)
                    n += 1
                    if compare_rankings(r.ranking, b)["exact_match"]:
                        ok += 1
                    else:
                        print("FAIL", seed, agg, missing, mode)
                    fc = full_scan_reads(d.sources)
                    sa_eng += r.ledger.total_sa
                    sa_full += fc["total_sa"]
                    if r.ledger.total_sa < fc["total_sa"]:
                        early += 1
    print(f"配置数 {n}，通过 {ok}，提前停止 {early}")
    print(f"SA 节省 {100 * (1 - sa_eng / sa_full):.1f}% "
          f"({sa_eng} vs {sa_full})")
    sys.exit(0 if ok == n else 1)


def cmd_drift(args) -> None:
    """查询进行中替换源数据，验证 continue/restart 都不拼版。"""
    for policy in ("continue", "restart"):
        _hr(f"漂移策略 on_drift={policy}")
        d2 = load("counterexample")
        # 记录旧版本全量基线（continue 应当与它一致）
        old_baseline = full_join(d2.sources, d2.spec, 3)
        eng = TopKEngine(
            d2.sources, d2.spec, QueryConfig(k=3, mode="ta", on_drift=policy)
        )
        eng.step()
        eng.step()
        # 查询中途把 sales 改成完全不同的数据
        target = d2.sources[0]
        old_v = target.current_version
        new_rows = [Row("zzz", 0.99, 0), Row("yyy", 0.98, 1)]
        target.replace(new_rows)
        print(f"替换 {target.name}: v{old_v} -> v{target.current_version}")
        result = eng.run()
        print(f"重启次数: {result.ledger.restarts}")
        drift_events = [
            e["drift"]["type"]
            for e in eng.ledger.trace if e.get("drift")
        ]
        print(f"漂移事件: {drift_events}")

        if policy == "continue":
            check = compare_rankings(result.ranking, old_baseline)
            print(f"continue: 与旧版本基线核对: "
                  f"{'精确一致 ✓（全程读旧快照，未拼版）' if check['exact_match'] else check}")
            print(f"引擎持有的 {target.name} 快照版本: "
                  f"v{eng.snaps[target.name].version}（当前实际 v{target.current_version}）")
        else:
            # restart：整次重开，结果必须等于新版本数据的全量基线
            new_baseline = full_join(d2.sources, d2.spec, 3)
            check = compare_rankings(result.ranking, new_baseline)
            print(f"restart: 与新版本基线核对: "
                  f"{'精确一致 ✓（重开后只用新版本，未拼版）' if check['exact_match'] else check}")
            print(f"引擎持有的 {target.name} 快照版本: "
                  f"v{eng.snaps[target.name].version}（= 当前 v{target.current_version}）")


def main() -> None:
    p = argparse.ArgumentParser(description="多源单调聚合精确 Top-K 演示")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("demo").set_defaults(func=cmd_demo)

    pr = sub.add_parser("run")
    pr.add_argument("dataset", choices=list(BUILTIN))
    pr.add_argument("--mode", choices=["ta", "nra"], default="ta")
    pr.add_argument("--k", type=int, default=3)
    pr.add_argument("--aggregator", choices=["wsum", "min", "max", "product"])
    pr.add_argument("--missing", choices=["neutral", "exclude"])
    pr.add_argument("--sa-policy", choices=["round_robin", "cheapest"],
                    default="round_robin")
    pr.add_argument("--on-drift", choices=["continue", "restart"], default="continue")
    pr.add_argument("--latency", type=float, default=0.0)
    pr.set_defaults(func=cmd_run)

    ps = sub.add_parser("stress")
    ps.add_argument("--seeds", type=int, default=60)
    ps.set_defaults(func=cmd_stress)

    sub.add_parser("drift").set_defaults(func=cmd_drift)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
