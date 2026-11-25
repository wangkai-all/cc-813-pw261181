"""HTTP API 层测试：用标准库 urllib 直接打真实服务（不 mock）。

运行: python -I tests/test_api.py
需要先启动服务: python -I webapp.py --port 8123
"""

import json
import sys
import urllib.request

sys.path.insert(0, "/workspace")

BASE = "http://127.0.0.1:8123"


def call(path: str, body: dict | None = None, method: str = "POST") -> dict:
    data = json.dumps(body or {}).encode("utf-8") if body is not None else None
    req = urllib.request.Request(
        BASE + path, data=data, headers={"Content-Type": "application/json"},
        method=method,
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> int:
    failures = []

    def check(cond: bool, msg: str) -> None:
        print(("  ok " if cond else "FAIL ") + msg)
        if not cond:
            failures.append(msg)

    # 1. 数据集清单
    ds = call("/api/datasets", method="GET")
    check(
        {d["name"] for d in ds["datasets"]}
        == {"counterexample", "one_to_many", "missing_keys", "ties"},
        "四个内置数据集可列出",
    )

    # 2. 显式 name 必须隔离不同数据集（回归：曾误用全局当前数据集）
    call("/api/load", {"name": "counterexample"})
    r_missing = call("/api/run", {"name": "missing_keys", "mode": "nra", "k": 3})
    keys = [x["key"] for x in r_missing["result"]["ranking"]]
    check(set(keys) == {"x", "y", "z"}, f"missing_keys 请求返回自己的键，实际 {keys}")

    r_otm = call("/api/run", {"name": "one_to_many", "mode": "ta", "k": 3})
    check(
        [x["key"] for x in r_otm["result"]["ranking"]] == ["q", "r", "p"],
        "one_to_many 请求返回自己的榜单",
    )

    # 3. 反例数据集的核心叙事
    r = call("/api/run", {"name": "counterexample", "mode": "ta", "k": 3})
    check(
        [x["key"] for x in r["result"]["ranking"]] == ["star", "mid1", "a1"],
        "TA 找到隐藏赢家 star",
    )
    check(r["correctness"]["exact_match"], "TA 与全量基线精确一致")
    check(r["result"]["stop_evidence"]["rule"] == "threshold_bound", "TA 可证明提前停止")
    check(r["savings"]["sa_saved_pct"] > 0, f"SA 有实际节省 {r['savings']['sa_saved_pct']}%")

    # 4. NRA 零随机访问
    r_nra = call("/api/run", {"name": "counterexample", "mode": "nra", "k": 3})
    check(r_nra["result"]["ledger"]["total_ra"] == 0, "NRA 不产生随机访问")
    check(r_nra["correctness"]["exact_match"], "NRA 与全量基线精确一致")

    # 5. 朴素局部 TopK 反例
    base = call("/api/baseline", {"name": "counterexample", "k": 3})
    missed = {m["key"] for m in base["naive_missed_winners"]}
    check({"star", "mid1"} <= missed, f"朴素拼接漏掉 star/mid1，实际漏 {missed}")

    # 6. 步进 + 漂移 continue
    sess = call("/api/step/start",
                {"name": "counterexample", "mode": "ta", "k": 3, "on_drift": "continue"})
    sid = sess["session_id"]
    step1 = call("/api/step/next", {"session_id": sid})
    check(step1["event"]["action"] == "sa", "步进返回一次排序访问事件")
    mut = call("/api/mutate", {"source": "sales"})
    check(mut["new_version"] == mut["old_version"] + 1, "突变使源版本 +1")
    step2 = call("/api/step/next", {"session_id": sid})
    check(
        step2["event"]["drift"] is not None
        and step2["event"]["drift"]["type"] == "drift_continue",
        "continue 策略产生 drift_continue 事件",
    )

    # 7. 漂移 restart
    sess2 = call("/api/step/start",
                 {"name": "counterexample", "mode": "ta", "k": 3, "on_drift": "restart"})
    call("/api/step/next", {"session_id": sess2["session_id"]})
    call("/api/mutate", {"source": "reviews"})
    step3 = call("/api/step/next", {"session_id": sess2["session_id"]})
    check(
        step3["event"]["drift"] is not None
        and step3["event"]["drift"]["type"] == "drift_restart",
        "restart 策略产生 drift_restart 事件",
    )

    # 8. 慢源模拟改变虚拟耗时但不改变答案（重新载入，避免前面突变的影响）
    call("/api/load", {"name": "counterexample"})
    r = call("/api/run", {"name": "counterexample", "mode": "ta", "k": 3})
    slow = call("/api/run",
                {"name": "counterexample", "mode": "ta", "k": 3,
                 "latencies": {"traffic": 100.0}})
    check(
        slow["result"]["ranking"] == r["result"]["ranking"],
        "模拟慢源不改变精确榜单",
    )
    check(
        slow["result"]["ledger"]["virtual_time"] > r["result"]["ledger"]["virtual_time"],
        "慢源延迟被计入虚拟耗时",
    )

    print(f"\n{len(failures)} 个失败" if failures else "\n全部 API 检查通过")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
