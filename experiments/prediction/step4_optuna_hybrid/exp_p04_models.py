"""EXP-P04 模型定义：LSTM / BiLSTM / CNN-LSTM / CNN-BiLSTM / MiniPatchTST / PatchTST。"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# LSTM
# ---------------------------------------------------------------------------

class LSTMRegressor(nn.Module):
    def __init__(self, n_features: int, horizon: int = 1, hidden: int = 64, layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            n_features, hidden, num_layers=layers, batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


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
# CNN-LSTM
# ---------------------------------------------------------------------------

class CNNLSTMRegressor(nn.Module):
    def __init__(self, n_features: int, horizon: int = 1, conv_channels: int = 32,
                 kernel_size: int = 3, lstm_hidden: int = 64, lstm_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, conv_channels, kernel_size=kernel_size, padding="same"),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(
            conv_channels, lstm_hidden, num_layers=lstm_layers, batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.head = nn.Linear(lstm_hidden, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
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
# MiniPatchTST (轻量 Transformer，作为 Optuna 候选)
# ---------------------------------------------------------------------------

class PatchEmbedding(nn.Module):
    def __init__(self, seq_len: int, patch_len: int, stride: int, d_model: int, n_features: int):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.n_features = n_features
        n_patches = (seq_len - patch_len) // stride + 1
        self.n_patches = n_patches
        self.proj = nn.Linear(patch_len * n_features, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        x = x.transpose(1, 2)
        patches = x.unfold(2, self.patch_len, self.stride)
        patches = patches.reshape(B, D * self.patch_len, self.n_patches)
        patches = patches.transpose(1, 2)
        patches = self.proj(patches)
        cls = self.cls_token.expand(B, -1, -1)
        patches = torch.cat([cls, patches], dim=1)
        return patches + self.pos_embed


class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x


class MiniPatchTSTRegressor(nn.Module):
    """轻量 PatchTST：d_model=32, n_heads=2, num_layers=1。"""
    def __init__(self, seq_len: int, n_features: int, horizon: int = 1,
                 patch_len: int = 4, stride: int = 2, d_model: int = 32,
                 n_heads: int = 2, num_layers: int = 1, dropout: float = 0.2):
        super().__init__()
        self.patch_embed = PatchEmbedding(seq_len, patch_len, stride, d_model, n_features)
        self.blocks = nn.ModuleList(
            [TransformerEncoderBlock(d_model, n_heads, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        cls_out = x[:, 0, :]
        return self.head(cls_out)


class PatchTSTRegressor(nn.Module):
    """标准 PatchTST（保留作为对照）。"""
    def __init__(self, seq_len: int, n_features: int, horizon: int = 1,
                 patch_len: int = 4, stride: int = 2, d_model: int = 64,
                 n_heads: int = 4, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.patch_embed = PatchEmbedding(seq_len, patch_len, stride, d_model, n_features)
        self.blocks = nn.ModuleList(
            [TransformerEncoderBlock(d_model, n_heads, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        cls_out = x[:, 0, :]
        return self.head(cls_out)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_model(name: str, n_features: int, seq_len: int, horizon: int = 1, **kwargs) -> nn.Module:
    name = name.lower().replace("-", "_")
    if name == "lstm":
        return LSTMRegressor(n_features=n_features, horizon=horizon, **kwargs)
    if name == "bilstm":
        return BiLSTMRegressor(n_features=n_features, horizon=horizon, **kwargs)
    if name == "cnn_lstm":
        return CNNLSTMRegressor(n_features=n_features, horizon=horizon, **kwargs)
    if name == "cnn_bilstm":
        return CNNBiLSTMRegressor(n_features=n_features, horizon=horizon, **kwargs)
    if name == "minipatchtst":
        return MiniPatchTSTRegressor(seq_len=seq_len, n_features=n_features, horizon=horizon, **kwargs)
    if name == "patchtst":
        return PatchTSTRegressor(seq_len=seq_len, n_features=n_features, horizon=horizon, **kwargs)
    raise ValueError(f"未知模型: {name}")
