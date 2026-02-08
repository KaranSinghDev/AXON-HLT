"""HIGGS dataset downloader — streams 100K rows from UCI, caches locally."""

import gzip
import io
import urllib.request
from pathlib import Path

import pandas as pd

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
