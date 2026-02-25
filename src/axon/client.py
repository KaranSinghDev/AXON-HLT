"""Triton gRPC client wrapper with basic retry logic."""

from __future__ import annotations

import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException


class TritonClient:
    def __init__(self, url: str = "localhost:8001", timeout: float = 5.0):
        self._client = grpcclient.InferenceServerClient(url=url, verbose=False)
        self._timeout = timeout

    def is_ready(self, model_name: str) -> bool:
        try:
            return self._client.is_model_ready(model_name)
        except Exception:
            return False

    def infer(self, model_name: str, data: np.ndarray) -> np.ndarray:
        """Send a single inference request, return output array."""
        inp = grpcclient.InferInput("input", data.shape, "FP32")
        inp.set_data_from_numpy(data)
        out = grpcclient.InferRequestedOutput("output")

        for attempt in range(3):
            try:
                result = self._client.infer(
                    model_name=model_name,
                    inputs=[inp],
                    outputs=[out],
                    client_timeout=self._timeout,
                )
                return result.as_numpy("output")
            except InferenceServerException as e:
                if attempt == 2:
                    raise
                continue

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> TritonClient:
        return self

    def __exit__(self, *_) -> None:
        self.close()
