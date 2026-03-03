"""Generate benchmark result plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

plt.rcParams.update({
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
})

_COLORS = {"batching_on": "#1f77b4", "batching_off": "#d62728", "v2": "#2ca02c"}


def plot_concurrency_sweep(records: list[dict], out: Path) -> None:
    """Throughput and p99 latency vs concurrency level."""
    concurrencies = [r["concurrency"] for r in records]
    throughputs = [r["throughput_inf_per_sec"] for r in records]
    p99s = [r["latency_p99_ms"] for r in records]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(concurrencies, throughputs, marker="o", color=_COLORS["batching_on"], linewidth=2)
    ax1.set_xlabel("Concurrent clients")
    ax1.set_ylabel("Throughput (inf / sec)")
    ax1.set_title("Throughput vs Concurrency")
    ax1.xaxis.set_major_locator(ticker.FixedLocator(concurrencies))

    ax2.plot(concurrencies, p99s, marker="s", color=_COLORS["batching_off"], linewidth=2)
    ax2.set_xlabel("Concurrent clients")
    ax2.set_ylabel("p99 latency (ms)")
    ax2.set_title("p99 Latency vs Concurrency")
    ax2.xaxis.set_major_locator(ticker.FixedLocator(concurrencies))

    model = records[0]["model_name"] if records else ""
    fig.suptitle(f"Concurrency sweep — {model}", fontsize=13, y=1.02)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_ab_comparison(on_record: dict, off_record: dict, out: Path) -> None:
    """Side-by-side bar chart: dynamic batching ON vs OFF."""
    labels = ["Throughput\n(inf/sec)", "p99 Latency\n(ms)"]
    on_vals = [on_record["throughput_inf_per_sec"], on_record["latency_p99_ms"]]
    off_vals = [off_record["throughput_inf_per_sec"], off_record["latency_p99_ms"]]

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))

    bars_on = ax.bar(x - width / 2, on_vals, width, label="Dynamic batching ON",
                     color=_COLORS["batching_on"], alpha=0.85)
    bars_off = ax.bar(x + width / 2, off_vals, width, label="Dynamic batching OFF",
                      color=_COLORS["batching_off"], alpha=0.85)

    ax.bar_label(bars_on, fmt="%.0f", padding=3, fontsize=9)
    ax.bar_label(bars_off, fmt="%.0f", padding=3, fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title(
        f"Dynamic Batching A/B — {on_record['model_name']} "
        f"(concurrency={on_record['concurrency']})",
        fontsize=12,
    )
    ax.legend()
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_latency_cdf(records: list[dict], labels: list[str], out: Path) -> None:
    """Latency CDF for one or more benchmark runs (requires raw latencies)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [_COLORS["batching_on"], _COLORS["batching_off"], _COLORS["v2"]]

    for record, label, color in zip(records, labels, colors):
        lats = record.get("latencies_ms", [])
        if not lats:
            continue
        sorted_lats = np.sort(lats)
        cdf = np.arange(1, len(sorted_lats) + 1) / len(sorted_lats)
        ax.plot(sorted_lats, cdf, label=label, color=color, linewidth=1.8)

    ax.axvline(x=7.0, color="gray", linestyle="--", linewidth=1, alpha=0.7,
               label="CERN HLT budget (7ms)")
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("CDF")
    ax.set_title("End-to-end latency distribution")
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.02)
    ax.legend()
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")
