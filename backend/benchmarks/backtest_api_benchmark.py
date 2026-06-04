#!/usr/bin/env python3
"""Simple API benchmark for Backtest dashboard endpoints.

Usage:
  python benchmarks/backtest_api_benchmark.py --base-url http://localhost:8000 --campaign-id <batch_id>
"""

import argparse
import statistics
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Dict, List


def _request(url: str, timeout: float) -> float:
    start = time.perf_counter()
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        response.read()
    end = time.perf_counter()
    return (end - start) * 1000.0


def _percentile(sorted_values: List[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * p
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = rank - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _summarize(values: List[float]) -> Dict[str, float]:
    sorted_values = sorted(values)
    return {
        "count": float(len(values)),
        "min": round(sorted_values[0], 2),
        "max": round(sorted_values[-1], 2),
        "mean": round(statistics.fmean(values), 2),
        "p50": round(_percentile(sorted_values, 0.50), 2),
        "p95": round(_percentile(sorted_values, 0.95), 2),
        "p99": round(_percentile(sorted_values, 0.99), 2),
    }


def _run_endpoint(name: str, url: str, iterations: int, timeout: float) -> Dict[str, float]:
    measurements = []
    failures = 0

    for _ in range(iterations):
        try:
            measurements.append(_request(url, timeout=timeout))
        except (urllib.error.URLError, TimeoutError) as exc:
            failures += 1
            print(f"[WARN] {name} request failed: {exc}")

    if not measurements:
        return {
            "count": 0,
            "failures": failures,
            "min": 0,
            "max": 0,
            "mean": 0,
            "p50": 0,
            "p95": 0,
            "p99": 0,
        }

    summary = _summarize(measurements)
    summary["failures"] = failures
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Backtest API latency (p50/p95/p99)")
    parser.add_argument("--base-url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--campaign-id", required=True, help="Batch ID used for detail/results/top/chart routes")
    parser.add_argument("--iterations", type=int, default=30, help="Requests per endpoint")
    parser.add_argument("--timeout", type=float, default=8.0, help="Per-request timeout seconds")
    parser.add_argument("--campaign-type", default="backtest", choices=["backtest", "wfa"], help="Campaign type for list/stats")
    args = parser.parse_args()

    base = args.base_url.rstrip("/")
    campaign_id = urllib.parse.quote(args.campaign_id, safe="")
    campaign_type = urllib.parse.quote(args.campaign_type, safe="")

    endpoints = {
        "stats": f"{base}/api/campaigns/stats?type={campaign_type}",
        "list": f"{base}/api/campaigns?type={campaign_type}&limit=100",
        "detail": f"{base}/api/campaigns/{campaign_id}",
        "results": f"{base}/api/campaigns/{campaign_id}/results?limit=100&skip=0&sort_by=roi&sort_order=-1",
        "top": f"{base}/api/campaigns/{campaign_id}/top-strategies?limit=20",
        "chart": f"{base}/api/campaigns/{campaign_id}/chart-data?points=1000",
    }

    print("Backtest API benchmark")
    print(f"Base URL: {base}")
    print(f"Campaign ID: {args.campaign_id}")
    print(f"Iterations per endpoint: {args.iterations}")
    print()

    for name, url in endpoints.items():
        summary = _run_endpoint(name, url, args.iterations, args.timeout)
        print(
            f"{name:8} count={int(summary['count']):2d} fail={int(summary['failures']):2d} "
            f"p50={summary['p50']:7.2f}ms p95={summary['p95']:7.2f}ms p99={summary['p99']:7.2f}ms "
            f"mean={summary['mean']:7.2f}ms"
        )


if __name__ == "__main__":
    main()
