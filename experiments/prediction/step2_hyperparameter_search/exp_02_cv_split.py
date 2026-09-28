"""EXP-P04 交叉验证工具。

提供滚动窗口交叉验证划分功能。
"""

from __future__ import annotations

import numpy as np


def create_rolling_folds(
    n_total: int,
    n_folds: int = 3,
    train_frac: float = 0.667,
    step: int | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """创建滚动窗口交叉验证划分。

    采用时间序列滚动窗口策略：
    - 窗口大小固定为 train_frac * n_total
    - 每个 fold 的训练集从窗口起始位置扩展到固定长度
    - 验证集为窗口后的下一个时间块

    Args:
        n_total: 总样本数
        n_folds: 折叠数（默认 3）
        train_frac: 训练集占总数据的比例（默认 0.667）
        step: 可选，每步滑动的样本数，默认自动计算

    Returns:
        list of (train_indices, val_indices) tuples
    """
    n_train = int(n_total * train_frac)
    n_val = n_total - n_train

    if step is None:
        # 均匀分布所有 folds
        step = n_val // n_folds

    folds = []
    for fold_idx in range(n_folds):
        # 训练集: 从起点到 train_frac 位置
        train_start = 0
        train_end = n_train

        # 验证集: 紧跟训练集之后
        val_start = train_end + fold_idx * step
        val_end = min(val_start + n_val, n_total)

        # 边界检查
        if val_start >= n_total:
            break

        train_indices = np.arange(train_start, train_end)
        val_indices = np.arange(val_start, val_end)

        folds.append((train_indices, val_indices))

    return folds


def create_time_series_cv(
    n_total: int,
    n_splits: int = 5,
    test_size: float = 0.2,
    gap: int = 0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """创建时间序列交叉验证划分（替代方案）。

    Args:
        n_total: 总样本数
        n_splits: 划分数量
        test_size: 测试集比例
        gap: 训练集和测试集之间的间隔（防止数据泄露）

    Returns:
        list of (train_indices, test_indices) tuples
    """
    test_len = max(1, int(n_total * test_size))
    n_train = n_total - test_len

    folds = []
    for i in range(n_splits):
        # 训练集: 0 到 n_train
        train_indices = np.arange(0, n_train)

        # 测试集: n_train + gap + i*offset
        offset = i * (test_len // n_splits)
        test_start = n_train + gap + offset
        test_end = min(test_start + test_len, n_total)

        if test_start >= n_total:
            break

        test_indices = np.arange(test_start, test_end)
        folds.append((train_indices, test_indices))

    return folds
