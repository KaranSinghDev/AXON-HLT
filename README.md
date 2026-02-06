# Axon

A coprocessor offloading framework — Python-based Proof-of-Concept of CERN's
**SONIC** (Services for Optimized Network Inference on Coprocessors) architecture.

Demonstrates high-throughput, low-latency ML inference using NVIDIA Triton Inference
Server, ONNX, and gRPC — the production stack deployed at CMS for the High-Level
Trigger (HLT).

## Why this exists

At the LHC, protons collide 40 million times per second. The High-Level Trigger
must filter these events to ~1,000/sec under a strict <7ms latency budget. Modern
ML triggers (CNNs, GNNs like ParticleNet) are too slow on CPUs and too expensive
to put one GPU per HLT node. CERN's solution: **decouple inference**. CPU nodes
send gRPC requests to a centralized Triton GPU cluster — that's SONIC.

This project distills SONIC's core architectural pattern into a clean, standalone
Python PoC, using the same production-grade tools (Triton, ONNX) that CMS deploys.

## Architecture

```
Build-Time
──────────
HIGGS dataset → train_model.py → particle_classifier.onnx

Run-Time
────────
benchmark_client.py    →  gRPC  →  Triton Inference Server
(asyncio load gen,                  (Docker, GPU)
 simulates HLT farm)               │
                       ←  results ─┤  loads model_repository/
                                   │    particle_classifier_v1/
                                   │      config.pbtxt  ← dynamic batching
                                   │      1/model.onnx
```

## Tech stack

| Component | Tool | Version |
|---|---|---|
| Inference server | NVIDIA Triton | 25.12-py3 |
| Model format | ONNX | 1.21 |
| ML framework | PyTorch | 2.11 (CPU) |
| Transport | gRPC | via `tritonclient[grpc]` 2.64 |
| Dataset | UCI HIGGS | 28 features, 100K-row subset |
| Concurrency | Python `asyncio` | — |

## Quickstart

```bash
make install         # create venv, install deps
make train           # download HIGGS, train MLP, export ONNX (~2 min)
make serve           # start Triton container
make benchmark       # run gRPC load test
make plot            # generate throughput/latency plots
```

## Status

🚧 In development.

## License

Apache 2.0 — see [LICENSE](LICENSE).

## References

- [CMS-MLG-23-001](https://arxiv.org/abs/2402.15366) — *Portable Acceleration of CMS Production Workflow with Coprocessors as a Service* (CMS Collaboration, Feb 2024)
- [NVIDIA Triton Inference Server](https://github.com/triton-inference-server/server)
- [UCI HIGGS Dataset](https://archive.ics.uci.edu/ml/datasets/HIGGS)
