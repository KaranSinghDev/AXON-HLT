"""Collect Triton server-side metrics from the Prometheus endpoint (port 8002)."""

from __future__ import annotations

import re
import urllib.request

_METRICS_URL = "http://localhost:8002/metrics"

_PATTERNS = {
    "gpu_utilization": re.compile(
        r'nv_gpu_utilization\{[^}]*\}\s+([\d.]+)'
    ),
    "gpu_memory_used_bytes": re.compile(
        r'nv_gpu_memory_used_bytes\{[^}]*\}\s+([\d.]+)'
    ),
    "inference_count": re.compile(
        r'nv_inference_count\{[^}]*model="([^"]+)"[^}]*\}\s+([\d.]+)'
    ),
    "inference_queue_us": re.compile(
        r'nv_inference_queue_duration_us\{[^}]*model="([^"]+)"[^}]*\}\s+([\d.]+)'
    ),
}


def scrape(url: str = _METRICS_URL) -> dict:
    """Fetch and parse Triton Prometheus metrics. Returns empty dict on failure."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "axon/0.1"})
        with urllib.request.urlopen(req, timeout=3) as resp:
            text = resp.read().decode("utf-8")
    except Exception:
        return {}

    result: dict = {}

    m = _PATTERNS["gpu_utilization"].search(text)
    if m:
        result["gpu_utilization_pct"] = float(m.group(1))

    m = _PATTERNS["gpu_memory_used_bytes"].search(text)
    if m:
        result["gpu_memory_used_mb"] = float(m.group(1)) / (1024 ** 2)

    counts: dict[str, float] = {}
    for m in _PATTERNS["inference_count"].finditer(text):
        counts[m.group(1)] = float(m.group(2))
    if counts:
        result["inference_counts"] = counts

    queues: dict[str, float] = {}
    for m in _PATTERNS["inference_queue_us"].finditer(text):
        queues[m.group(1)] = float(m.group(2))
    if queues:
        result["inference_queue_us"] = queues

    return result
