"""Tests for axon.export."""

import tempfile
from pathlib import Path

import numpy as np
import onnx
import pytest
import torch

from axon.export import export_onnx, validate_onnx
from axon.model import build_v1, build_v2

N_FEATURES = 28


@pytest.fixture
def v1_onnx(tmp_path):
    model = build_v1(N_FEATURES)
    path = tmp_path / "v1.onnx"
    export_onnx(model, N_FEATURES, path)
    return path, model


@pytest.fixture
def v2_onnx(tmp_path):
    model = build_v2(N_FEATURES)
    path = tmp_path / "v2.onnx"
    export_onnx(model, N_FEATURES, path)
    return path, model


def test_export_creates_file(v1_onnx):
    path, _ = v1_onnx
    assert path.exists()
    assert path.stat().st_size > 0


def test_exported_model_is_valid_onnx(v1_onnx):
    path, _ = v1_onnx
    proto = onnx.load(str(path))
    onnx.checker.check_model(proto)


def test_batch_dim_is_dynamic(v1_onnx):
    path, _ = v1_onnx
    proto = onnx.load(str(path))
    input_shape = proto.graph.input[0].type.tensor_type.shape
    batch_dim = input_shape.dim[0]
    assert batch_dim.dim_param, "batch dim must be symbolic for Triton dynamic batching"


def test_feature_dim_is_fixed(v1_onnx):
    path, _ = v1_onnx
    proto = onnx.load(str(path))
    input_shape = proto.graph.input[0].type.tensor_type.shape
    feature_dim = input_shape.dim[1].dim_value
    assert feature_dim == N_FEATURES


def test_validate_onnx_with_model(v1_onnx):
    path, model = v1_onnx
    validate_onnx(path, N_FEATURES, model=model)


def test_pytorch_onnx_output_agreement(v1_onnx):
    path, model = v1_onnx
    import onnxruntime as ort

    model.eval()
    x_np = np.random.randn(8, N_FEATURES).astype(np.float32)
    with torch.no_grad():
        ref = model(torch.from_numpy(x_np)).numpy()

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    ort_out = sess.run(None, {"input": x_np})[0]
    np.testing.assert_allclose(ref, ort_out, atol=1e-4)


def test_v2_also_exports_cleanly(v2_onnx):
    path, model = v2_onnx
    validate_onnx(path, N_FEATURES, model=model)
