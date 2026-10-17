"""
PyTorch 训练工具: DataLoader、带早停的训练循环、批量预测。

注意: 不要用 Apple GPU (MPS) 训练, 实测 PyTorch 2.4 在 MPS 上训练该模型不收敛。
"""

import copy
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(y, dtype=torch.float32))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def run_epoch(model, loader, device, optimizer=None, desc=None) -> float:
    """跑一个 epoch, 返回平均 MSE。传入 optimizer 时训练, 否则只计算损失。"""
    training = optimizer is not None
    model.train(training)
    loss_fn = nn.MSELoss()
    total, n = 0.0, 0
    batches = tqdm(loader, desc=desc, unit="batch", leave=False, dynamic_ncols=True) if desc else loader
    with torch.set_grad_enabled(training):
        for X, y in batches:
            X, y = X.to(device), y.to(device)
            loss = loss_fn(model(X), y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += loss.item()
            n += 1
            if desc:
                batches.set_postfix(loss=f"{total / n:.5f}")
    return total / max(n, 1)


def train_with_early_stop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    lr: float,
    max_epochs: int = 20,
    patience: int = 5,
    device: torch.device | None = None,
    progress: bool = False,
) -> tuple[nn.Module, list[dict]]:
    """Adam + ReduceLROnPlateau, 验证损失连续 patience 个 epoch 不下降就停止, 返回验证集最优的权重。"""
    device = device or get_device()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    history, best_loss, best_state, no_improve = [], float("inf"), None, 0
    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        desc = f"Epoch {epoch}/{max_epochs}" if progress else None
        train_loss = run_epoch(model, train_loader, device, optimizer, desc)
        val_loss = run_epoch(model, val_loader, device)
        scheduler.step(val_loss)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                        "lr": optimizer.param_groups[0]["lr"], "time_sec": time.time() - t0})

        improved = val_loss < best_loss
        if improved:
            best_loss, best_state, no_improve = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            no_improve += 1
        if progress:
            tqdm.write(f"Epoch {epoch:>2}/{max_epochs}  train_loss={train_loss:.5f}  val_loss={val_loss:.5f}  "
                       f"lr={history[-1]['lr']:.1e}  {history[-1]['time_sec']:.1f}s"
                       + ("  * best" if improved else f"  (未提升 {no_improve}/{patience})"))
        if no_improve >= patience:
            if progress:
                tqdm.write(f"早停: 验证损失连续 {patience} 个 epoch 未提升")
            break

    model.load_state_dict(best_state)
    return model, history


def predict(model: nn.Module, X: np.ndarray, device: torch.device | None = None, batch_size: int = 512) -> np.ndarray:
    device = device or get_device()
    model = model.to(device).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            out.append(model(torch.as_tensor(X[i:i + batch_size], dtype=torch.float32, device=device)).cpu().numpy())
    return np.concatenate(out, axis=0)
