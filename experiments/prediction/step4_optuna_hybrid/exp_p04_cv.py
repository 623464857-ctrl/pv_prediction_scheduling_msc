"""EXP-P04 Rolling-Origin 验证（时间顺序，不 shuffle）。"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


def create_rolling_folds(
    n_total: int,
    n_folds: int = 3,
    train_frac: float = 2 / 3,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    按时间顺序构造 rolling folds。
    每折：前 train_frac 训练，后 (1 - train_frac) 验证。
    返回 list of (train_idx, val_idx)，均为 int64 ndarray。
    """
    fold_size = n_total // n_folds
    folds = []
    for i in range(n_folds):
        fold_start = i * fold_size
        fold_end = (i + 1) * fold_size if i < n_folds - 1 else n_total

        fold_mid = fold_start + int((fold_end - fold_start) * train_frac)
        train_idx = np.arange(fold_start, fold_mid, dtype=np.int64)
        val_idx = np.arange(fold_mid, fold_end, dtype=np.int64)
        folds.append((train_idx, val_idx))

    return folds
