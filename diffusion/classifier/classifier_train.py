#!/usr/bin/env python3
"""
Time‑series RNN classifier
-------------------------
• Expected input  shape : (batch, seq_len, n_features)
• Predicted output shape: (batch,) **or** (batch, n_classes) depending on `num_classes`

This script defines:
1. `TimeSeriesDataset`  – a light‑weight `torch.utils.data.Dataset` that wraps
   numpy arrays or tensors.
2. `RNNClassifier`      – a configurable LSTM/GRU backbone followed by a fully
   connected layer that maps the last hidden state to class logits.
3. Training / evaluation loops with a command‑line interface for easy
   experimentation.

Example (binary classification with dummy data):
$ python timeseries_rnn_classifier.py --num_classes 1 --epochs 10
"""
from __future__ import annotations

import argparse
from pathlib import Path
from classifier.model import RNNClassifier
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import os
import pandas as pd
from typing import Optional, Tuple


class TimeSeriesDataset(Dataset):

    def __init__(
        self,
        data,
        labels,
        normalize=True,
        stats=None,
        eps=1e-8,
    ):

        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        if isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels)
        assert data.ndim == 3, "data must be (N, seq_len, n_features)"
        assert len(data) == len(labels)
        self.data = data.float()
        self.labels = labels.long()
        self.normalize = normalize
        self.eps = eps

        if self.normalize:
            if stats is None:
                # compute mean/std over all time‑steps *per feature*
                mean = self.data.mean(dim=(0, 1), keepdim=True)  # (1,1,F)
                std = self.data.std(dim=(0, 1), keepdim=True)
            else:
                mean, std = stats
                if isinstance(mean, np.ndarray):
                    mean = torch.from_numpy(mean)
                if isinstance(std, np.ndarray):
                    std = torch.from_numpy(std)
                mean, std = mean.float(), std.float()
            self.register_buffer("_mean",
                                 mean)  # cached on device when .to(...)
            self.register_buffer("_std", std.clamp_min(self.eps))

    # tiny helper so buffers exist even on CPU tensors
    def register_buffer(self, name: str, tensor: torch.Tensor):
        object.__setattr__(self, name, tensor)

    # expose stats to reuse on other splits
    @property
    def stats(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if not self.normalize:
            return None
        return self._mean, self._std

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        if self.normalize:
            x = (x - self._mean) / self._std
        x = x.squeeze(0)
        return x, self.labels[idx]


# -----------------------------------------------------------------------------
# Train / Eval helpers
# -----------------------------------------------------------------------------


def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    if logits.dim() == 1:  # binary with BCEWithLogits
        preds = (torch.sigmoid(logits) > 0.5).long()
    else:  # multi‑class with CE
        preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


def _run_epoch(model, loader, criterion, optimizer, device, train=True):
    if train:
        model.train()
    else:
        model.eval()
    total_loss = 0.0
    total_acc = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if train:
            optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y.float() if logits.dim() == 1 else y)
        if train:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * x.size(0)
        total_acc += _accuracy(logits, y) * x.size(0)
    n = len(loader.dataset)
    return total_loss / n, total_acc / n


# -----------------------------------------------------------------------------
# Main – quick demo on synthetic data
# -----------------------------------------------------------------------------


def main(args):
    rng = np.random.default_rng(args.seed)
    if os.path.exists(args.train_data) and os.path.exists(args.val_data):
        X_train, y_train = pd.read_pickle(args.train_data)
        X_train = X_train.transpose(0, 2, 1)
        X_val, y_val = pd.read_pickle(args.val_data)
        X_val = X_val.transpose(0, 2, 1)
    else:
        X_train = rng.standard_normal(size=(20000, 24, 7), dtype=np.float32)
        y_train = rng.integers(0,
                               args.num_classes,
                               size=(20000, ),
                               dtype=np.int64)
        X_val = rng.standard_normal(size=(5000, 24, 7), dtype=np.float32)
        y_val = rng.integers(0,
                             args.num_classes,
                             size=(5000, ),
                             dtype=np.int64)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train_set = TimeSeriesDataset(X_train, y_train)
    val_set = TimeSeriesDataset(X_val, y_val)

    train_loader = DataLoader(train_set,
                              batch_size=args.batch_size,
                              shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RNNClassifier(
        input_dim=7,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        rnn_type=args.rnn_type,
        num_classes=args.num_classes,
        dropout=args.dropout,
    ).to(device)

    criterion = (nn.BCEWithLogitsLoss()
                 if args.num_classes == 1 else nn.CrossEntropyLoss())
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = 0.0
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)

    for epoch in tqdm(range(1, args.epochs + 1)):
        tr_loss, tr_acc = _run_epoch(model,
                                     train_loader,
                                     criterion,
                                     optimizer,
                                     device,
                                     train=True)
        va_loss, va_acc = _run_epoch(model,
                                     val_loader,
                                     criterion,
                                     optimizer,
                                     device,
                                     train=False)
        print(
            f"Epoch {epoch:02d} | train {tr_loss:.4f}/{tr_acc:.4f} | val {va_loss:.4f}/{va_acc:.4f}"
        )
        if va_acc > best_val:
            best_val = va_acc
            print(f"New best val acc: {best_val:.4f} -> saving model")
            torch.save({"model_state": model.state_dict()},
                       Path(args.ckpt_dir) / "best_model.pt")
    print(f"Train Finished. Best val acc: {best_val:.4f}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Bidirectional LSTM/GRU time‑series classifier")
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--rnn_type", choices=["lstm", "gru"], default="lstm")
    p.add_argument("--num_classes",
                   type=int,
                   default=1,
                   help="1 for binary, >1 for multi‑class")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--train_data", type=str, default="data/train_data.npy")
    p.add_argument("--val_data", type=str, default="data/val_data.npy")
    p.add_argument("--ckpt_dir", type=str, default="checkpoints")
    p.add_argument("--seed", type=int, default=42)

    args = p.parse_args()
    main(args)
