"""Generate plots from saved benchmark JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from axon.plot import plot_ab_comparison, plot_concurrency_sweep, plot_latency_cdf

RESULTS_DIR = Path("results")
PLOTS_DIR = RESULTS_DIR / "plots"


def load_json(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    return data if isinstance(data, list) else [data]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate plots from benchmark results")
    p.add_argument("--sweep", type=Path, default=None,
                   help="JSON from --sweep run (concurrency sweep)")
    p.add_argument("--batching-on", type=Path, default=None,
                   help="JSON from batching-enabled benchmark")
    p.add_argument("--batching-off", type=Path, default=None,
                   help="JSON from no-batching benchmark")
    p.add_argument("--output-dir", type=Path, default=PLOTS_DIR)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    generated = 0

    if args.sweep:
        records = load_json(args.sweep)
        if len(records) > 1:
            plot_concurrency_sweep(records, args.output_dir / "concurrency_sweep.png")
            generated += 1
        else:
            print(f"Skipping sweep plot: only {len(records)} record(s) in {args.sweep}")

    if args.batching_on and args.batching_off:
        on_records = load_json(args.batching_on)
        off_records = load_json(args.batching_off)
        on = on_records[0]
        off = off_records[0]
        plot_ab_comparison(on, off, args.output_dir / "ab_comparison.png")
        generated += 1

        if on.get("latencies_ms") and off.get("latencies_ms"):
            plot_latency_cdf(
                [on, off],
                ["Batching ON", "Batching OFF"],
                args.output_dir / "latency_cdf.png",
            )
            generated += 1

    if generated == 0:
        print("No plots generated. Pass --sweep, or both --batching-on and --batching-off.")
        print("\nExamples:")
        print("  python scripts/plot_results.py --sweep results/benchmark_sweep.json")
        print("  python scripts/plot_results.py \\")
        print("    --batching-on  results/benchmark_batching_on.json \\")
        print("    --batching-off results/benchmark_batching_off.json")
    else:
        print(f"\n{generated} plot(s) saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
