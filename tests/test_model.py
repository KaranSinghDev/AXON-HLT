"""Tests for axon.model."""

import torch
import pytest

from axon.model import MLP, build_v1, build_v2

N_FEATURES = 28
BATCH = 16


def test_v1_output_shape():
    model = build_v1(N_FEATURES)
    x = torch.randn(BATCH, N_FEATURES)
    out = model(x)
    assert out.shape == (BATCH, 1)


def test_v2_output_shape():
    model = build_v2(N_FEATURES)
    x = torch.randn(BATCH, N_FEATURES)
    out = model(x)
    assert out.shape == (BATCH, 1)


def test_v1_fewer_params_than_v2():
    v1_params = sum(p.numel() for p in build_v1().parameters())
    v2_params = sum(p.numel() for p in build_v2().parameters())
    assert v1_params < v2_params


def test_batch_size_one():
    model = build_v1(N_FEATURES)
    x = torch.randn(1, N_FEATURES)
    out = model(x)
    assert out.shape == (1, 1)


def test_no_sigmoid_in_final_layer():
    # model outputs raw logits — sigmoid is applied by BCEWithLogitsLoss
    model = build_v1(N_FEATURES)
    last = list(model.net.children())[-1]
    assert isinstance(last, torch.nn.Linear)


def test_custom_architecture():
    model = MLP([28, 256, 128, 1])
    x = torch.randn(4, 28)
    assert model(x).shape == (4, 1)
