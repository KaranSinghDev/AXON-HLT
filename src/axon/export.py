"""Export trained PyTorch models to ONNX for Triton deployment."""

from pathlib import Path

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
    )
