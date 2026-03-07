"""Run inference benchmark against a live Triton server and save results to JSON."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from axon.benchmark import BenchmarkConfig, run
from axon.client import TritonClient
from axon.metrics import scrape

CONCURRENCY_SWEEP = [1, 4, 16, 64, 128, 256]
DEFAULT_MODEL = "particle_classifier_v1"
RESULTS_DIR = Path("results")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Axon benchmark client")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--url", default="localhost:8001")
    p.add_argument("--concurrency", type=int, default=64)
    p.add_argument("--duration", type=float, default=30.0, help="seconds")
    p.add_argument("--warmup", type=float, default=5.0, help="seconds")
    p.add_argument("--sweep", action="store_true", help="run full concurrency sweep")
    p.add_argument("--output", type=Path, default=None)
    return p.parse_args()


def wait_for_server(url: str, model: str, timeout: int = 60) -> None:
    print(f"Waiting for Triton at {url} ...", end="", flush=True)
    with TritonClient(url) as c:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if c.is_ready(model):
                print(" ready.")
                return
            time.sleep(2)
            print(".", end="", flush=True)
    raise TimeoutError(f"Triton not ready after {timeout}s")


def print_result(result) -> None:
    print(f"\n{'─'*50}")
    print(f"  model       : {result.model_name}")
    print(f"  concurrency : {result.concurrency}")
    print(f"  inferences  : {result.total_inferences:,}")
    print(f"  throughput  : {result.throughput:,.1f} inf/sec")
    print(f"  latency p50 : {result.p50_ms:.2f} ms")
    print(f"  latency p95 : {result.p95_ms:.2f} ms")
    print(f"  latency p99 : {result.p99_ms:.2f} ms")
    print(f"  latency mean: {result.mean_ms:.2f} ms")
    print(f"{'─'*50}")


def result_to_dict(result, server_metrics: dict) -> dict:
    return {
        "model_name": result.model_name,
        "concurrency": result.concurrency,
        "duration_secs": round(result.duration_secs, 2),
        "total_inferences": result.total_inferences,
        "throughput_inf_per_sec": round(result.throughput, 1),
        "latency_p50_ms": round(result.p50_ms, 3),
        "latency_p95_ms": round(result.p95_ms, 3),
        "latency_p99_ms": round(result.p99_ms, 3),
        "latency_mean_ms": round(result.mean_ms, 3),
        "latencies_ms": [round(x, 3) for x in result.latencies_ms],
        "server_metrics": server_metrics,
    }


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(exist_ok=True)

    wait_for_server(args.url, args.model)

    configs = (
        [BenchmarkConfig(model_name=args.model, url=args.url, concurrency=c,
                         duration_secs=args.duration, warmup_secs=args.warmup)
         for c in CONCURRENCY_SWEEP]
        if args.sweep
        else [BenchmarkConfig(model_name=args.model, url=args.url,
                              concurrency=args.concurrency,
                              duration_secs=args.duration, warmup_secs=args.warmup)]
    )

    all_results = []
    for cfg in configs:
        print(f"\nRunning: model={cfg.model_name}  concurrency={cfg.concurrency}"
              f"  duration={cfg.duration_secs}s  warmup={cfg.warmup_secs}s")
        metrics_before = scrape()
        result = asyncio.run(run(cfg))
        metrics_after = scrape()
        print_result(result)
        if metrics_after.get("gpu_utilization_pct") is not None:
            print(f"  GPU util    : {metrics_after['gpu_utilization_pct']:.1f}%")
        all_results.append(result_to_dict(result, metrics_after))

    out_path = args.output or RESULTS_DIR / f"benchmark_{args.model}_{int(time.time())}.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
