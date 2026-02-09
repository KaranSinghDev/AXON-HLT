"""HIGGS dataset: downloader and preprocessing pipeline."""

import gzip
import io
import urllib.request
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz"
_COLS = ["label"] + [f"f{i}" for i in range(1, 29)]
_NROWS = 100_000


def download_higgs(dest: Path, nrows: int = _NROWS) -> pd.DataFrame:
    """Stream HIGGS.csv.gz from UCI, stop after nrows, cache as CSV."""
    dest = Path(dest)
    if dest.exists():
        return pd.read_csv(dest)

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {nrows:,} rows from UCI HIGGS dataset...")

    rows: list[str] = []
    req = urllib.request.Request(_URL, headers={"User-Agent": "axon/0.1"})
    with urllib.request.urlopen(req) as resp:
        with gzip.GzipFile(fileobj=resp) as gz:
            for i, raw in enumerate(gz):
                if i >= nrows:
                    break
                rows.append(raw.decode("utf-8").strip())

    df = pd.read_csv(
        io.StringIO("\n".join(rows)),
        header=None,
        names=_COLS,
    )
    df.to_csv(dest, index=False)
    print(f"Cached {len(df):,} rows to {dest}")
    return df


class DataSplit(NamedTuple):
    X_train: np.ndarray
    X_val: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    scaler: StandardScaler
    n_features: int


def load_dataset(path: Path, val_frac: float = 0.2, seed: int = 42) -> DataSplit:
    """Load cached HIGGS CSV, scale features, return stratified train/val split."""
    df = pd.read_csv(path)
    X = df.drop(columns=["label"]).values.astype(np.float32)
    y = df["label"].values.astype(np.float32)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_frac, random_state=seed, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)

    return DataSplit(X_train, X_val, y_train, y_val, scaler, n_features=X.shape[1])
