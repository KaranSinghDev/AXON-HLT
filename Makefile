.PHONY: help install train serve serve-no-batching stop benchmark benchmark-baseline sweep plot test clean

PYTHON := venv/bin/python
PIP := venv/bin/pip

help:
	@echo "Axon — SONIC-style Inference-as-a-Service PoC"
	@echo ""
	@echo "Targets:"
	@echo "  install              Create venv and install dependencies"
	@echo "  train                Download HIGGS subset and train MLP, export ONNX"
	@echo "  serve                Start Triton with dynamic batching"
	@echo "  serve-no-batching    Start Triton without batching (A/B baseline)"
	@echo "  stop                 Stop Triton container"
	@echo "  benchmark            Run benchmark against running Triton (default config)"
	@echo "  benchmark-baseline   Run benchmark against no-batching server"
	@echo "  sweep                Run concurrency sweep benchmark"
	@echo "  plot                 Generate plots from results/"
	@echo "  test                 Run pytest"
	@echo "  clean                Remove generated artifacts"

install:
	python3 -m venv venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

train:
	$(PYTHON) scripts/train.py

serve:
	docker compose up -d triton

serve-no-batching:
	docker compose -f docker-compose.no-batching.yml up -d triton

stop:
	docker compose down || true
	docker compose -f docker-compose.no-batching.yml down || true

benchmark:
	$(PYTHON) scripts/benchmark.py --config configs/benchmark.json --output results/benchmark_batching_on.json

benchmark-baseline:
	$(PYTHON) scripts/benchmark.py --config configs/benchmark.json --output results/benchmark_batching_off.json

sweep:
	$(PYTHON) scripts/benchmark.py --sweep --output results/benchmark_sweep.json

plot:
	$(PYTHON) scripts/plot_results.py --input results/ --output results/plots/

test:
	$(PYTHON) -m pytest tests/ -v

clean:
	rm -rf data/ results/*.json results/plots/*.png logs/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
