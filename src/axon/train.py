"""Training loop for HEP particle classifier."""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from axon.data import DataSplit


@dataclass
class TrainConfig:
    epochs: int = 10
    batch_size: int = 512
    lr: float = 1e-3
    device: str = "cpu"


@dataclass
class TrainResult:
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    val_accuracies: list[float] = field(default_factory=list)


def train(model: nn.Module, split: DataSplit, cfg: TrainConfig) -> TrainResult:
    device = torch.device(cfg.device)
    model = model.to(device)

    X_tr = torch.from_numpy(split.X_train).to(device)
    y_tr = torch.from_numpy(split.y_train).unsqueeze(1).to(device)
    X_val = torch.from_numpy(split.X_val).to(device)
    y_val = torch.from_numpy(split.y_val).unsqueeze(1).to(device)

    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=cfg.batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.BCEWithLogitsLoss()
    result = TrainResult()

    for epoch in range(cfg.epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in tqdm(loader, desc=f"epoch {epoch+1}/{cfg.epochs}", leave=False):
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        result.train_losses.append(epoch_loss / len(split.X_train))

        model.eval()
        with torch.no_grad():
            logits = model(X_val)
            val_loss = criterion(logits, y_val).item()
            preds = (torch.sigmoid(logits) > 0.5).float()
            acc = (preds == y_val).float().mean().item()
        result.val_losses.append(val_loss)
        result.val_accuracies.append(acc)
        print(f"  epoch {epoch+1}: val_loss={val_loss:.4f}  val_acc={acc:.4f}")

    return result
