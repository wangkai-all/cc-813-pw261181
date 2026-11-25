"""网页服务启动入口（零第三方依赖，标准库 http.server）。"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from leaderboard.app import serve  # noqa: E402

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="多源单调聚合 Top-K 分析查询网页")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()
    serve(args.host, args.port)
