"""Asyncio load generator — simulates the HLT CPU farm sending concurrent gRPC requests."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

import numpy as np
import tritonclient.grpc.aio as async_grpc
from tritonclient.utils import InferenceServerException


@dataclass
class BenchmarkConfig:
    model_name: str = "particle_classifier_v1"
    url: str = "localhost:8001"
    n_features: int = 28
    concurrency: int = 64
    duration_secs: float = 30.0
    warmup_secs: float = 5.0
    batch_size: int = 1


@dataclass
class BenchmarkResult:
    model_name: str
    concurrency: int
    duration_secs: float
    total_inferences: int
    throughput: float          # inferences / second
    latencies_ms: list[float] = field(default_factory=list)

    @property
    def p50_ms(self) -> float:
        return float(np.percentile(self.latencies_ms, 50)) if self.latencies_ms else 0.0

    @property
    def p95_ms(self) -> float:
        return float(np.percentile(self.latencies_ms, 95)) if self.latencies_ms else 0.0

    @property
    def p99_ms(self) -> float:
        return float(np.percentile(self.latencies_ms, 99)) if self.latencies_ms else 0.0

    @property
    def mean_ms(self) -> float:
        return float(np.mean(self.latencies_ms)) if self.latencies_ms else 0.0


async def _worker(
    client: async_grpc.InferenceServerClient,
    cfg: BenchmarkConfig,
    start_time: float,
    warmup_until: float,
    stop_event: asyncio.Event,
    latencies: list[float],
    counter: list[int],
) -> None:
    rng = np.random.default_rng()
    while not stop_event.is_set():
        data = rng.random((cfg.batch_size, cfg.n_features), dtype=np.float32)
        inp = async_grpc.InferInput("input", data.shape, "FP32")
        inp.set_data_from_numpy(data)
        out = async_grpc.InferRequestedOutput("output")

        t0 = time.perf_counter()
        try:
            await client.infer(
                model_name=cfg.model_name,
                inputs=[inp],
                outputs=[out],
            )
        except (InferenceServerException, Exception):
            continue
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        now = time.perf_counter()
        if now >= warmup_until:
            latencies.append(elapsed_ms)
            counter[0] += cfg.batch_size


async def run(cfg: BenchmarkConfig) -> BenchmarkResult:
    """Drive cfg.concurrency async workers against Triton for cfg.duration_secs."""
    client = async_grpc.InferenceServerClient(url=cfg.url, verbose=False)
    latencies: list[float] = []
    counter = [0]
    stop_event = asyncio.Event()

    start = time.perf_counter()
    warmup_until = start + cfg.warmup_secs
    measure_until = start + cfg.warmup_secs + cfg.duration_secs

    workers = [
        asyncio.create_task(
            _worker(client, cfg, start, warmup_until, stop_event, latencies, counter)
        )
        for _ in range(cfg.concurrency)
    ]

    await asyncio.sleep(cfg.warmup_secs + cfg.duration_secs)
    stop_event.set()
    await asyncio.gather(*workers, return_exceptions=True)
    await client.close()

    actual_duration = time.perf_counter() - warmup_until
    throughput = counter[0] / actual_duration if actual_duration > 0 else 0.0

    return BenchmarkResult(
        model_name=cfg.model_name,
        concurrency=cfg.concurrency,
        duration_secs=actual_duration,
        total_inferences=counter[0],
        throughput=throughput,
        latencies_ms=latencies,
    )
