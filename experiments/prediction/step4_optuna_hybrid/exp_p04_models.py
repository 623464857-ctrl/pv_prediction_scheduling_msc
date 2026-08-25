"""EXP-P04 模型定义：BiLSTM / CNN-BiLSTM（残差预测）。"""

from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# BiLSTM
# ---------------------------------------------------------------------------

class BiLSTMRegressor(nn.Module):
    def __init__(self, n_features: int, horizon: int = 1, hidden: int = 64, layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            n_features, hidden, num_layers=layers, batch_first=True,
            dropout=dropout if layers > 1 else 0.0, bidirectional=True,
        )
        self.head = nn.Linear(hidden * 2, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


# ---------------------------------------------------------------------------
# CNN-BiLSTM
# ---------------------------------------------------------------------------

class CNNBiLSTMRegressor(nn.Module):
    def __init__(self, n_features: int, horizon: int = 1, conv_channels: int = 32,
                 kernel_size: int = 3, bilstm_hidden: int = 64, bilstm_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, conv_channels, kernel_size=kernel_size, padding="same"),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(
            conv_channels, bilstm_hidden, num_layers=bilstm_layers, batch_first=True,
            dropout=dropout if bilstm_layers > 1 else 0.0, bidirectional=True,
        )
        self.head = nn.Linear(bilstm_hidden * 2, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_model(name: str, n_features: int, seq_len: int, horizon: int = 1, **kwargs) -> nn.Module:
    name = name.lower().replace("-", "_")
    if name == "bilstm":
        return BiLSTMRegressor(n_features=n_features, horizon=horizon, **kwargs)
    if name == "cnn_bilstm":
        return CNNBiLSTMRegressor(n_features=n_features, horizon=horizon, **kwargs)
    raise ValueError(f"未知模型: {name}，仅支持 bilstm / cnn_bilstm")
