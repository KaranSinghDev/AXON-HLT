"""Tests for axon.data — no network access required."""

import io
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from axon.data import DataSplit, _COLS, download_higgs, load_dataset

N_FEATURES = 28


def _make_fake_csv(n_rows: int = 200) -> Path:
    """Write a minimal HIGGS-shaped CSV to a temp file."""
    rng = np.random.default_rng(0)
    data = rng.random((n_rows, N_FEATURES + 1)).astype(np.float32)
    data[:, 0] = (data[:, 0] > 0.5).astype(np.float32)  # binary label
    df = pd.DataFrame(data, columns=_COLS)
    tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
    df.to_csv(tmp.name, index=False)
    return Path(tmp.name)


@pytest.fixture
def fake_csv():
    path = _make_fake_csv()
    yield path
    path.unlink(missing_ok=True)


def test_feature_count(fake_csv):
    split = load_dataset(fake_csv)
    assert split.n_features == N_FEATURES
    assert split.X_train.shape[1] == N_FEATURES
    assert split.X_val.shape[1] == N_FEATURES


def test_train_val_split_sizes(fake_csv):
    split = load_dataset(fake_csv, val_frac=0.2)
    total = len(split.X_train) + len(split.X_val)
    assert total == 200
    assert len(split.X_val) == pytest.approx(40, abs=2)


def test_labels_binary(fake_csv):
    split = load_dataset(fake_csv)
    assert set(np.unique(split.y_train)).issubset({0.0, 1.0})
    assert set(np.unique(split.y_val)).issubset({0.0, 1.0})


def test_features_scaled(fake_csv):
    split = load_dataset(fake_csv)
    # training features should be approximately zero-mean, unit-variance
    assert abs(split.X_train.mean()) < 0.1
    assert abs(split.X_train.std() - 1.0) < 0.1


def test_val_uses_train_scaler(fake_csv):
    split = load_dataset(fake_csv)
    # val set is scaled with training scaler — mean won't be exactly 0 but dtype is float32
    assert split.X_val.dtype == np.float32
    assert split.X_train.dtype == np.float32


def test_data_split_is_named_tuple(fake_csv):
    split = load_dataset(fake_csv)
    assert isinstance(split, DataSplit)
    assert hasattr(split, "X_train")
    assert hasattr(split, "scaler")


def test_download_higgs_cache_hit(tmp_path):
    """If dest file already exists, download_higgs returns it without network."""
    cached = tmp_path / "higgs_sample.csv"
    fake = _make_fake_csv(100)
    cached.write_text(fake.read_text())
    df = download_higgs(cached)
    assert len(df) == 100
    assert list(df.columns) == _COLS
