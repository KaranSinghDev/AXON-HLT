"""Train both MLP variants and export each to ONNX."""

import argparse
import json
from pathlib import Path

import torch

from axon.data import download_higgs, load_dataset
from axon.export import export_onnx
from axon.model import build_v1, build_v2
from axon.train import TrainConfig, train

DATA_DIR = Path("data")
MODEL_DIR = Path("model_repository")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--nrows", type=int, default=100_000)
    return p.parse_args()


def main():
    args = parse_args()
    DATA_DIR.mkdir(exist_ok=True)

    print("=== Preparing data ===")
    df = download_higgs(DATA_DIR / "higgs_sample.csv", nrows=args.nrows)
    split = load_dataset(DATA_DIR / "higgs_sample.csv")
    n_feat = split.n_features
    print(f"Train: {len(split.X_train):,}  Val: {len(split.X_val):,}  Features: {n_feat}")

    cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size)
    variants = {"particle_classifier_v1": build_v1(n_feat), "particle_classifier_v2": build_v2(n_feat)}

    for name, model in variants.items():
        print(f"\n=== Training {name} ===")
        result = train(model, split, cfg)
        print(f"Final val_acc: {result.val_accuracies[-1]:.4f}")

        out_path = MODEL_DIR / name / "1" / "model.onnx"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        export_onnx(model, n_features=n_feat, path=out_path)
        print(f"Exported {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
