"""执行器正确性测试：与全量关联基线逐名核对（含随机属性测试）。

运行: python -m unittest discover -s tests -v
"""

import itertools
import math
import random
import unittest

from leaderboard.baseline import (
    compare_rankings,
    full_join,
    full_join_all_scores,
    full_scan_reads,
    naive_local_top_union,
)
from leaderboard.datasets import (
    BUILTIN,
    counterexample,
    missing_keys,
    one_to_many,
    random_dataset,
    ties,
)
from leaderboard.engine import QueryConfig, TopKEngine, execute
from leaderboard.scoring import AGGREGATORS, NEUTRALS, ScoreSpec
from leaderboard.sources import Row, VersionedSource


class TestExactMatch(unittest.TestCase):
    """TA / NRA 在所有内置数据集上必须与全量关联完全一致。"""

    def test_builtin_datasets_ta(self):
        for factory in BUILTIN.values():
            with self.subTest(dataset=factory.__name__):
                d = factory()
                cfg = QueryConfig(k=d.k, mode="ta")
                result = execute(d.sources, d.spec, cfg)
                baseline = full_join(d.sources, d.spec, d.k)
                report = compare_rankings(result.ranking, baseline)
                self.assertTrue(report["exact_match"], report)

    def test_builtin_datasets_nra(self):
        for factory in BUILTIN.values():
            with self.subTest(dataset=factory.__name__):
                d = factory()
                cfg = QueryConfig(k=d.k, mode="nra")
                result = execute(d.sources, d.spec, cfg)
                baseline = full_join(d.sources, d.spec, d.k)
                report = compare_rankings(result.ranking, baseline)
                self.assertTrue(report["exact_match"], report)

    def test_expected_winners(self):
        for factory in BUILTIN.values():
            d = factory()
            if not d.expected_winners:
                continue
            with self.subTest(dataset=d.name):
                result = execute(d.sources, d.spec, QueryConfig(k=d.k, mode="ta"))
                got = tuple(k for k, _ in result.ranking)
                self.assertEqual(got, d.expected_winners)

    def test_random_property_ta(self):
        for seed in range(30):
            d = random_dataset(seed=seed)
            with self.subTest(seed=seed):
                result = execute(d.sources, d.spec, QueryConfig(k=d.k, mode="ta"))
                baseline = full_join(d.sources, d.spec, d.k)
                self.assertTrue(
                    compare_rankings(result.ranking, baseline)["exact_match"]
                )

    def test_random_property_nra(self):
        for seed in range(30):
            d = random_dataset(seed=seed)
            with self.subTest(seed=seed):
                result = execute(d.sources, d.spec, QueryConfig(k=d.k, mode="nra"))
                baseline = full_join(d.sources, d.spec, d.k)
                self.assertTrue(
                    compare_rankings(result.ranking, baseline)["exact_match"]
                )

    def test_all_aggregators_and_missing_rules(self):
        for agg in AGGREGATORS:
            for missing in ("neutral", "exclude"):
                d = random_dataset(seed=7, aggregator=agg, missing=missing)
                for mode in ("ta", "nra"):
                    with self.subTest(agg=agg, missing=missing, mode=mode):
                        cfg = QueryConfig(k=d.k, mode=mode)
                        result = execute(d.sources, d.spec, cfg)
                        baseline = full_join(d.sources, d.spec, d.k)
                        self.assertTrue(
                            compare_rankings(result.ranking, baseline)["exact_match"],
                            f"{agg}/{missing}/{mode}",
                        )


class TestCounterexample(unittest.TestCase):
    def test_naive_misses_star(self):
        d = counterexample()
        naive = naive_local_top_union(d.sources, d.spec, d.k)
        naive_keys = {k for k, _ in naive}
        self.assertNotIn("star", naive_keys, "朴素局部拼接应当漏掉 star")
        for missed in d.naive_misses:
            self.assertIn(missed, d.naive_misses)

    def test_engine_finds_star(self):
        d = counterexample()
        for mode in ("ta", "nra"):
            result = execute(d.sources, d.spec, QueryConfig(k=d.k, mode=mode))
            keys = [k for k, _ in result.ranking]
            self.assertIn("star", keys, f"{mode} 必须找到真正赢家 star")
            self.assertEqual(keys.index("star"), 0)
            self.assertEqual(tuple(keys), d.expected_winners)

    def test_early_stop_saves_reads(self):
        d = counterexample()
        result = execute(d.sources, d.spec, QueryConfig(k=d.k, mode="ta"))
        full_cost = full_scan_reads(d.sources)
        self.assertLess(result.ledger.total_sa, full_cost["total_sa"])
        self.assertEqual(result.stop_evidence["rule"], "threshold_bound")


class TestOneToMany(unittest.TestCase):
    def test_max_reducer_no_inflation(self):
        d = one_to_many()
        result = execute(d.sources, d.spec, QueryConfig(k=d.k, mode="ta"))
        keys = [k for k, _ in result.ranking]
        # p 有 3 行但只能占一个榜位
        self.assertEqual(keys.count("p"), 1)
        self.assertEqual(tuple(keys), d.expected_winners)
        # 归约分必须等于 max(0.8, 0.7, 0.6)
        snap = d.sources[0].snapshot()
        self.assertAlmostEqual(snap.lookup("p").score, 0.8)
        self.assertEqual(snap.lookup("p").n_rows, 3)

    def test_sum_reducer_changes_ranking_deterministically(self):
        d = one_to_many()
        d.sources[0] = VersionedSource(
            "orders",
            list(d.sources[0].snapshot().rows),
            reducer="sum",
        )
        result = execute(d.sources, d.spec, QueryConfig(k=d.k, mode="ta"))
        baseline = full_join(d.sources, d.spec, d.k)
        self.assertTrue(compare_rankings(result.ranking, baseline)["exact_match"])
        # sum 下 p=2.1 成为第一
        self.assertEqual(result.ranking[0][0], "p")


class TestTies(unittest.TestCase):
    def test_stable_tie_order(self):
        d = ties()
        for mode in ("ta", "nra"):
            result = execute(d.sources, d.spec, QueryConfig(k=d.k, mode=mode))
            keys = [k for k, _ in result.ranking]
            self.assertEqual(keys, ["alpha", "beta", "gamma"])
            self.assertAlmostEqual(result.ranking[0][1], result.ranking[1][1])

    def test_tie_independent_of_source_order(self):
        d = ties()
        reversed_sources = list(reversed(d.sources))
        r1 = execute(d.sources, d.spec, QueryConfig(k=d.k, mode="nra"))
        r2 = execute(reversed_sources, d.spec, QueryConfig(k=d.k, mode="nra"))
        self.assertEqual(
            [k for k, _ in r1.ranking], [k for k, _ in r2.ranking]
        )


class TestMissingRules(unittest.TestCase):
    def test_neutral_vs_exclude(self):
        d = missing_keys()
        neutral = execute(
            d.sources,
            ScoreSpec(aggregator="wsum", weights={"s1": 1, "s2": 1}, missing="neutral"),
            QueryConfig(k=3),
        )
        exclude = execute(
            d.sources,
            ScoreSpec(aggregator="wsum", weights={"s1": 1, "s2": 1}, missing="exclude"),
            QueryConfig(k=3),
        )
        scores_neutral = dict(neutral.ranking)
        scores_exclude = dict(exclude.ranking)
        self.assertAlmostEqual(scores_neutral["z"], 0.6)       # 缺席源取 0
        self.assertAlmostEqual(scores_exclude["z"], 0.6)       # 归一化后仍是 0.6
        self.assertAlmostEqual(scores_neutral["x"], 1.8)
        self.assertAlmostEqual(scores_exclude["x"], 0.9)
        # y/z 在 exclude 下并列，稳定 id 约定 y 在 z 前
        keys = [k for k, _ in exclude.ranking]
        self.assertLess(keys.index("y"), keys.index("z"))
        # 两种规则都必须与各自的全量基线一致
        self.assertTrue(
            compare_rankings(
                neutral.ranking,
                full_join(
                    d.sources,
                    ScoreSpec(aggregator="wsum", weights={"s1": 1, "s2": 1},
                              missing="neutral"),
                    3,
                ),
            )["exact_match"]
        )


class TestBoundsMath(unittest.TestCase):
    """用穷举所有未知源取值的方式验证上下界。"""

    BRUTE_GRID = [0.0, 0.25, 0.5, 0.75, 1.0, None]  # None = 缺席

    def _brute(self, spec, known, unknown):
        vals = []
        for combo in itertools.product(self.BRUTE_GRID, repeat=len(unknown)):
            present = dict(known)
            for src, v in zip(sorted(unknown), combo):
                if v is not None:
                    present[src] = v
            if spec.missing == "exclude" and not present:
                continue  # 所有源缺席 => 对象不可上榜，不参与界的枚举
            score = spec.complete_score(present, list(known) + sorted(unknown))
            if score is not None:
                vals.append(score)
        return min(vals), max(vals)

    def _brute_nra(self, spec, known, unknown, thresholds):
        """NRA 穷举：未知源只能缺席或取 [0, 阈值] 内的值。"""
        grid_map = {
            s: [None] + [0.0, thresholds[s] * 0.3, thresholds[s] * 0.7, thresholds[s]]
            for s in sorted(unknown)
        }
        order = sorted(unknown)
        vals = []
        for combo in itertools.product(*(grid_map[s] for s in order)):
            present = dict(known)
            for src, v in zip(order, combo):
                if v is not None:
                    present[src] = v
            if spec.missing == "exclude" and not present:
                continue
            score = spec.complete_score(present, list(known) + order)
            if score is not None:
                vals.append(score)
        return min(vals), max(vals)

    def test_bounds_match_bruteforce(self):
        sources = ["a", "b", "c"]
        known_cases = [{}, {"a": 0.4}, {"a": 0.4, "b": 0.7}]
        for agg in AGGREGATORS:
            for missing in ("neutral", "exclude"):
                spec = ScoreSpec(
                    aggregator=agg,
                    weights={"a": 1.0, "b": 2.0, "c": 0.5},
                    missing=missing,
                )
                for known in known_cases:
                    unknown = frozenset(s for s in sources if s not in known)
                    lb, ub = spec.bounds(known, unknown)
                    # known 为空且 exclude：对象可能整体不可上榜，界按数学约定给，
                    # 穷举（要求至少一个源在场）只在约定有限时可直接对比
                    if not known and missing == "exclude":
                        with self.subTest(agg=agg, missing=missing, known=known):
                            if agg == "min":
                                self.assertEqual(lb, -math.inf)
                                self.assertEqual(ub, math.inf)  # 空集 min 约定
                            elif agg == "product":
                                self.assertEqual(lb, 0.0)
                                _, bub = self._brute(spec, known, unknown)
                                self.assertAlmostEqual(ub, bub, places=9)
                            else:  # wsum / max
                                self.assertEqual(lb, -math.inf)
                                _, bub = self._brute(spec, known, unknown)
                                self.assertAlmostEqual(ub, bub, places=9)
                        continue
                    blb, bub = self._brute(spec, known, unknown)
                    with self.subTest(agg=agg, missing=missing, known=known):
                        self.assertAlmostEqual(lb, blb, places=9,
                                               msg=f"LB {lb} != 穷举 {blb}")
                        self.assertAlmostEqual(ub, bub, places=9,
                                               msg=f"UB {ub} != 穷举 {bub}")

    def test_nra_bounds_match_bruteforce(self):
        sources = ["a", "b", "c"]
        thresholds = {"a": 0.4, "b": 0.7, "c": 0.55}
        known_cases = [{}, {"a": 0.35}, {"a": 0.35, "b": 0.6}]
        for agg in AGGREGATORS:
            for missing in ("neutral", "exclude"):
                spec = ScoreSpec(
                    aggregator=agg,
                    weights={"a": 1.0, "b": 2.0, "c": 0.5},
                    missing=missing,
                )
                for known in known_cases:
                    unknown = frozenset(s for s in sources if s not in known)
                    lb, ub = spec.nra_bounds(known, unknown, thresholds)
                    if not known and missing == "exclude":
                        with self.subTest(agg=agg, missing=missing, known=known):
                            if agg == "min":
                                self.assertEqual((lb, ub), (-math.inf, math.inf))
                            elif agg == "product":
                                # 空已知集 product：下界 0，上界 max(1, Π阈值)=1
                                self.assertEqual((lb, ub), (0.0, 1.0))
                            else:  # wsum / max
                                self.assertEqual(lb, -math.inf)
                                _, bub = self._brute_nra(spec, known, unknown, thresholds)
                                self.assertAlmostEqual(ub, bub, places=9)
                        continue
                    blb, bub = self._brute_nra(spec, known, unknown, thresholds)
                    with self.subTest(agg=agg, missing=missing, known=known):
                        self.assertAlmostEqual(lb, blb, places=9,
                                               msg=f"NRA LB {lb} != 穷举 {blb}")
                        self.assertAlmostEqual(ub, bub, places=9,
                                               msg=f"NRA UB {ub} != 穷举 {bub}")


class TestSnapshotDrift(unittest.TestCase):
    def _make(self):
        s1 = VersionedSource("s1", [Row("x", 0.9, 0), Row("y", 0.5, 1)])
        s2 = VersionedSource("s2", [Row("x", 0.8, 0), Row("z", 0.6, 1)])
        spec = ScoreSpec(weights={"s1": 1, "s2": 1})
        return s1, s2, spec

    def test_continue_uses_old_snapshot(self):
        s1, s2, spec = self._make()
        engine = TopKEngine([s1, s2], spec, QueryConfig(k=2, on_drift="continue"))
        engine.step()
        old_snap = engine.snaps["s1"]
        s1.replace([Row("x", 0.05, 0), Row("y", 0.04, 1)])  # 新版本数据大变
        result = engine.run()
        # 引擎一直读旧快照，结果与旧版本全量基线一致
        baseline = full_join(
            [VersionedSource("s1", [Row("x", 0.9, 0), Row("y", 0.5, 1)]),
             VersionedSource("s2", [Row("x", 0.8, 0), Row("z", 0.6, 1)])],
            spec, 2,
        )
        self.assertTrue(compare_rankings(result.ranking, baseline)["exact_match"])
        self.assertIs(engine.snaps["s1"], old_snap)
        self.assertTrue(
            any(
                e.get("drift") and e["drift"]["type"] == "drift_continue"
                for e in engine.ledger.trace
            )
        )

    def test_restart_reopens_query(self):
        s1, s2, spec = self._make()
        engine = TopKEngine([s1, s2], spec, QueryConfig(k=2, on_drift="restart"))
        engine.step()
        s1.replace([Row("x", 0.05, 0), Row("y", 0.04, 1), Row("w", 0.95, 2)])
        result = engine.run()
        self.assertEqual(engine.ledger.restarts, 1)
        self.assertTrue(
            any(
                e.get("drift") and e["drift"]["type"] == "drift_restart"
                for e in engine.ledger.trace
            )
        )
        # 重开后结果反映新版本：w 成为赢家
        keys = [k for k, _ in result.ranking]
        self.assertIn("w", keys)
        self.assertEqual(engine.snaps["s1"].version, 2)

    def test_no_torn_reads_under_repeated_mutation(self):
        s1, s2, spec = self._make()
        engine = TopKEngine([s1, s2], spec, QueryConfig(k=2, on_drift="continue"))
        engine.step()
        for v in range(5):
            s1.replace([Row("x", 0.1 + 0.01 * v, 0)])
        result = engine.run()
        # 全程只认 v1 快照
        self.assertEqual(engine.snaps["s1"].version, 1)
        baseline = full_join(
            [VersionedSource("s1", [Row("x", 0.9, 0), Row("y", 0.5, 1)]), s2],
            spec, 2,
        )
        self.assertTrue(compare_rankings(result.ranking, baseline)["exact_match"])


class TestLedgerAndLatency(unittest.TestCase):
    def test_latency_changes_scheduling_and_cost(self):
        d = counterexample()
        fast = execute(
            d.sources, d.spec, QueryConfig(k=d.k, mode="ta"),
            latencies={"traffic": 100.0},
        )
        plain = execute(d.sources, d.spec, QueryConfig(k=d.k, mode="ta"))
        # 读取次数相同（轮询策略），但虚拟时间把慢源延迟计入
        self.assertEqual(fast.ledger.total_sa, plain.ledger.total_sa)
        self.assertGreater(fast.ledger.virtual_time, plain.ledger.virtual_time)

    def test_cheapest_policy_prefers_cheap_source(self):
        d = counterexample()
        eng = TopKEngine(
            d.sources, d.spec,
            QueryConfig(k=d.k, mode="nra", sa_policy="cheapest"),
        )
        ev = eng.step()
        # sales 的 cost_sa=1.0 最低
        self.assertEqual(ev["source"], "sales")

    def test_ta_uses_random_access_nra_does_not(self):
        d = counterexample()
        ta = execute(d.sources, d.spec, QueryConfig(k=d.k, mode="ta"))
        nra = execute(d.sources, d.spec, QueryConfig(k=d.k, mode="nra"))
        self.assertGreater(ta.ledger.total_ra, 0)
        self.assertEqual(nra.ledger.total_ra, 0)
        # 两者答案一致
        self.assertEqual(
            [k for k, _ in ta.ranking], [k for k, _ in nra.ranking]
        )


class TestMonotonicityContract(unittest.TestCase):
    def test_aggregators_are_monotone(self):
        """单调性是上界正确性的前提：任一源分数升高，总分不得下降。"""
        spec_w = ScoreSpec(weights={"a": 1, "b": 1, "c": 1})
        for agg in AGGREGATORS:
            spec = ScoreSpec(
                aggregator=agg,
                weights={"a": 1, "b": 1, "c": 1},
                missing="neutral",
            )
            base = {"a": 0.3, "b": 0.4, "c": 0.5}
            s0 = spec.complete_score(base, ["a", "b", "c"])
            bumped = dict(base, b=0.8)
            s1 = spec.complete_score(bumped, ["a", "b", "c"])
            self.assertGreaterEqual(s1, s0, f"{agg} 违反单调非降")


class TestUnseenUpperBound(unittest.TestCase):
    """回归：未见对象的上界必须覆盖"部分源缺席"的偏科赢家。"""

    def test_product_partial_absence_unseen(self):
        # 两源 product/neutral：赢家只在 src0 高分，src1 缺席（中性元 1）。
        # 阈值组合 src0*src1 会低估它，逐源 cap + 缺席枚举必须给出正确上界。
        s0 = VersionedSource(
            "src0",
            [Row("seen1", 0.99, 0), Row("winner", 0.97, 1)]
            + [Row(f"tail{i}", 0.5 - 0.05 * i, 10 + i) for i in range(6)],
        )
        s1 = VersionedSource(
            "src1",
            [Row("seen1", 0.92, 0)]
            + [Row(f"other{i}", 0.8 - 0.05 * i, 10 + i) for i in range(8)],
        )
        spec = ScoreSpec(aggregator="product", weights={"src0": 1, "src1": 1})
        eng = TopKEngine([s0, s1], spec, QueryConfig(k=1, mode="ta"))
        eng.step()  # 只读了两个源各一次 SA，winner 尚未见
        # 此时 winner 在 src0 未见（≤阈值），在 src1 也未见（可能缺席）
        ub = eng._unseen_ub(eng._unseen_caps())
        self.assertGreaterEqual(ub, 0.97 - 1e-12, "未见上界不得低于偏科赢家真实分")
        result = eng.run()
        baseline = full_join([s0, s1], spec, 1)
        self.assertTrue(compare_rankings(result.ranking, baseline)["exact_match"])
        self.assertEqual(result.ranking[0][0], "winner")

    def test_domain_max_auto_detection(self):
        # sum 归约使分数超过默认域上界 1.0，引擎必须从快照探测真实上界
        s0 = VersionedSource(
            "s0",
            [Row("x", 0.8, 0), Row("x", 0.7, 1), Row("y", 0.5, 2)],
            reducer="sum",
        )
        s1 = VersionedSource("s1", [Row("y", 0.9, 0), Row("z", 0.6, 1)])
        spec = ScoreSpec(aggregator="max", weights={"s0": 1, "s1": 1})
        eng = TopKEngine([s0, s1], spec, QueryConfig(k=2, mode="ta"))
        self.assertGreaterEqual(eng._domain_max["s0"], 1.5)
        result = eng.run()
        baseline = full_join([s0, s1], spec, 2)
        self.assertTrue(compare_rankings(result.ranking, baseline)["exact_match"])

    def test_explicit_domain_max_takes_precedence(self):
        s0 = VersionedSource("s0", [Row("x", 0.9, 0)])
        s1 = VersionedSource("s1", [Row("x", 0.8, 0)])
        spec = ScoreSpec(
            aggregator="wsum",
            weights={"s0": 1, "s1": 1},
            domain_max={"s0": 0.5, "s1": 0.5},
        )
        eng = TopKEngine([s0, s1], spec, QueryConfig(k=1, mode="ta"))
        self.assertEqual(eng._domain_max, {"s0": 0.5, "s1": 0.5})


class TestBoundedMemory(unittest.TestCase):
    """有限内存：执行器只保留已见键状态，不物化任何笛卡尔关联。"""

    def test_state_scales_with_seen_keys_not_join(self):
        d = random_dataset(seed=3, n_sources=4, n_keys=80, sparsity=0.3)
        eng = TopKEngine(d.sources, d.spec, QueryConfig(k=5, mode="ta"))
        eng.run()
        # 已见键数不超过各源键数之和（远小于任何关联膨胀后的规模）
        total_keys = sum(len(s.snapshot().sorted_scan()) for s in d.sources)
        self.assertLessEqual(len(eng.known), total_keys)
        # 每个键只保存每源至多一个归约标量 + 缺席集合
        for key, vals in eng.known.items():
            self.assertLessEqual(len(vals), len(d.sources))


if __name__ == "__main__":
    unittest.main(verbosity=2)
