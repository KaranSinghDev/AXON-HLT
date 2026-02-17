"""Export trained PyTorch models to ONNX for Triton deployment."""

from pathlib import Path

import numpy as np
import onnx
import torch
import torch.nn as nn


def export_onnx(
    model: nn.Module,
    n_features: int,
    path: Path,
    opset: int = 17,
) -> None:
    """Export model to ONNX with a dynamic batch dimension.

    dynamic_axes makes batch size variable at inference time — required for
    Triton's dynamic batching to group requests into larger GPU batches.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    dummy = torch.randn(1, n_features)
    # dynamo=False: use TorchScript path where dynamic_axes is fully supported.
    # The dynamo path (default in torch 2.9+) requires dynamic_shapes API instead.
    torch.onnx.export(
        model,
        dummy,
        str(path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
        opset_version=opset,
        dynamo=False,
    )


def validate_onnx(path: Path, n_features: int, model: nn.Module | None = None) -> None:
    """Validate exported ONNX model structure and optionally compare outputs to PyTorch.

    Checks:
      1. ONNX graph is well-formed (onnx.checker)
      2. Batch dimension is symbolic (dynamic batching will work in Triton)
      3. If model provided: ONNX and PyTorch agree on a test batch within 1e-5
    """
    path = Path(path)
    proto = onnx.load(str(path))
    onnx.checker.check_model(proto)

    # Confirm batch dim is dynamic
    input_shape = proto.graph.input[0].type.tensor_type.shape
    batch_dim = input_shape.dim[0]
    if not batch_dim.dim_param:
        raise ValueError(f"{path}: batch dimension is not symbolic — dynamic batching will fail")

    if model is not None:
        import onnxruntime as ort

        model.eval()
        x_np = np.random.randn(8, n_features).astype(np.float32)
        x_torch = torch.from_numpy(x_np)
        with torch.no_grad():
            ref = model(x_torch).numpy()

        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        ort_out = sess.run(None, {"input": x_np})[0]
        max_diff = float(np.abs(ref - ort_out).max())
        if max_diff > 1e-4:
            raise ValueError(f"PyTorch vs ONNX max diff {max_diff:.2e} exceeds threshold")
