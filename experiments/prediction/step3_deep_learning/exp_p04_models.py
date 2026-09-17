"""EXP-P04 深度学习模型定义。

本模块仅包含 CNN-BiLSTM 模型。
"""

import torch
import torch.nn as nn


class CNN_BiLSTM(nn.Module):
    """CNN-BiLSTM 残差预测模型。

    架构:
        Input -> CNN(1D) -> Dropout -> BiLSTM -> Dropout -> FC -> Output
    """

    def __init__(
        self,
        n_features: int,
        seq_len: int,
        horizon: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        cnn_channels: list[int] | tuple[int, ...] = (32, 64),
        kernel_size: int = 3,
    ):
        super().__init__()
        self.n_features = n_features
        self.seq_len = seq_len
        self.horizon = horizon

        # CNN blocks
        cnn_layers = []
        in_ch = n_features
        for out_ch in cnn_channels:
            cnn_layers.append(
                nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size, padding=kernel_size // 2)
            )
            cnn_layers.append(nn.BatchNorm1d(out_ch))
            cnn_layers.append(nn.ReLU())
            cnn_layers.append(nn.Dropout(dropout))
            in_ch = out_ch
        self.cnn = nn.Sequential(*cnn_layers)

        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=cnn_channels[-1] if cnn_channels else n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Fully connected
        lstm_output_size = hidden_size * 2  # bidirectional
        self.fc = nn.Linear(lstm_output_size, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features)
        x = x.permute(0, 2, 1)  # (batch, n_features, seq_len)
        x = self.cnn(x)  # (batch, last_cnn_ch, seq_len)
        x = x.permute(0, 2, 1)  # (batch, seq_len, last_cnn_ch)
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden*2)
        out = self.fc(lstm_out)  # (batch, seq_len, horizon)
        return out


def build_model(model_name: str, **kwargs) -> nn.Module:
    """构建模型实例。

    Args:
        model_name: 模型名称，目前仅支持 "cnn_bilstm"
        **kwargs: 模型参数

    Returns:
        模型实例
    """
    if model_name == "cnn_bilstm":
        return CNN_BiLSTM(**kwargs)
    else:
        raise ValueError(f"Unknown model: {model_name}. Only 'cnn_bilstm' is supported.")
