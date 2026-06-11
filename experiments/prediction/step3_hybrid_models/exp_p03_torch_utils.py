"""EXP-P03 PyTorch 训练工具（复用并扩展自 exp_p02_torch_utils）。"""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    X_t = torch.from_numpy(X.astype(np.float32))
    y_t = torch.from_numpy(y.astype(np.float32)).unsqueeze(1)
    return DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=shuffle)


def run_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    total, n = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(xb)
        n += len(xb)
    return total / max(n, 1)


@torch.no_grad()
def eval_loss(model, loader, criterion, device) -> float:
    model.eval()
    total, n = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        loss = criterion(pred, yb)
        total += loss.item() * len(xb)
        n += len(xb)
    return total / max(n, 1)


@torch.no_grad()
def predict(model, X: np.ndarray, device, batch_size: int = 512) -> np.ndarray:
    model.eval()
    loader = make_loader(X, np.zeros(len(X), dtype=np.float32), batch_size=batch_size, shuffle=False)
    preds = []
    for xb, _ in loader:
        preds.append(model(xb.to(device)).cpu().numpy())
    return np.concatenate(preds, axis=0).reshape(-1)


def train_with_early_stop(
    model,
    train_loader,
    val_loader,
    *,
    lr: float = 1e-3,
    max_epochs: int = 50,
    patience: int = 8,
    device=None,
) -> tuple[nn.Module, list[dict]]:
    device = device or get_device()
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_state = deepcopy(model.state_dict())
    best_val = float("inf")
    wait = 0
    history: list[dict] = []

    for epoch in range(1, max_epochs + 1):
        tr = run_epoch(model, train_loader, criterion, optimizer, device)
        va = eval_loss(model, val_loader, criterion, device)
        history.append({"epoch": epoch, "train_loss": tr, "val_loss": va})
        if va < best_val - 1e-6:
            best_val = va
            best_state = deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    return model, history
