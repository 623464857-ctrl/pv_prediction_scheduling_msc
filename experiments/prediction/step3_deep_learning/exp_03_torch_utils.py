"""EXP-P04 PyTorch 训练工具。"""

import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def get_device() -> torch.device:
    """获取可用的计算设备。"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def make_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 64,
    shuffle: bool = True,
) -> DataLoader:
    """从 numpy 数组创建 PyTorch DataLoader。

    Args:
        X: 特征数组
        y: 目标数组
        batch_size: 批次大小
        shuffle: 是否打乱

    Returns:
        DataLoader 实例
    """
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y)
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """训练一个epoch。

    Args:
        model: 模型
        train_loader: 训练数据加载器
        optimizer: 优化器
        device: 计算设备

    Returns:
        平均训练损失
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        y_pred = model(X_batch)

        # 处理目标形状 - y_pred 可能是 (batch, seq, horizon) 或 (batch, horizon)
        if y_pred.dim() == 3:
            # (batch, seq, horizon) -> 取最后一个时间步
            y_pred = y_pred[:, -1, :]

        loss = nn.MSELoss()(y_pred, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def eval_loss(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> float:
    """评估验证损失。

    Args:
        model: 模型
        val_loader: 验证数据加载器
        device: 计算设备

    Returns:
        平均验证损失
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            y_pred = model(X_batch)

            # 处理目标形状 - y_pred 可能是 (batch, seq, horizon) 或 (batch, horizon)
            if y_pred.dim() == 3:
                y_pred = y_pred[:, -1, :]

            loss = nn.MSELoss()(y_pred, y_batch)
            total_loss += loss.item()
            n_batches += 1

    return total_loss / max(n_batches, 1)


def train_with_early_stop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    lr: float = 0.001,
    max_epochs: int = 50,
    patience: int = 10,
    device: torch.device = None,
) -> tuple[nn.Module, list[dict]]:
    """带早停的训练循环。

    Args:
        model: 模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        lr: 学习率
        max_epochs: 最大epoch数
        patience: 早停耐心值
        device: 计算设备

    Returns:
        (最佳模型, 训练历史)
    """
    if device is None:
        device = get_device()
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    history = []
    best_val_loss = float("inf")
    best_model_state = None
    no_improve = 0

    for epoch in range(max_epochs):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = eval_loss(model, val_loader, device)
        elapsed = time.time() - t0

        scheduler.step(val_loss)

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": optimizer.param_groups[0]["lr"],
            "time_sec": elapsed,
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            break

    # 恢复最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, history


def predict(
    model: nn.Module,
    X: np.ndarray,
    device: torch.device = None,
    batch_size: int = 256,
) -> np.ndarray:
    """用模型进行批量预测。

    Args:
        model: 模型
        X: 输入数据 (N, seq_len, n_features)
        device: 计算设备
        batch_size: 批处理大小

    Returns:
        预测结果 (N, horizon)
    """
    if device is None:
        device = get_device()
    model = model.to(device)
    model.eval()

    X_tensor = torch.FloatTensor(X)
    predictions = []

    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            batch = X_tensor[i : i + batch_size].to(device)
            y_pred = model(batch)

            # 取最后一个时间步的输出
            if y_pred.dim() == 3:
                y_pred = y_pred[:, -1, :]

            predictions.append(y_pred.cpu().numpy())

    return np.concatenate(predictions, axis=0)
