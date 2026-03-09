FROM python:3.10-slim

WORKDIR /axon

COPY src/ src/
COPY scripts/ scripts/

RUN pip install --no-cache-dir \
    "tritonclient[grpc]==2.64.0" \
    "numpy>=1.24" \
    "matplotlib>=3.7" \
    "pandas>=2.0"

ENV PYTHONPATH=/axon/src

ENTRYPOINT ["python3", "scripts/benchmark.py"]
CMD ["--help"]
