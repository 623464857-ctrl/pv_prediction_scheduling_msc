"""
CNN-BiLSTM 光伏功率预测模型

    输入 (batch, L, 24) -> Conv1d x2 -> BiLSTM -> Linear -> 取最后时间步 -> 输出 (batch, H)

查看每一层的 tensor 形状:
python -m experiments.prediction.models --horizon 1
"""

import torch
import torch.nn as nn


class CNN_BiLSTM(nn.Module):
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

        # CNN: 每层 Conv1d -> BatchNorm -> ReLU -> Dropout, padding 保持时间长度不变
        cnn_layers = []
        in_ch = n_features
        for out_ch in cnn_channels:
            cnn_layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_ch = out_ch
        self.cnn = nn.Sequential(*cnn_layers)

        # 双向 LSTM: 输出维度 hidden_size * 2
        self.lstm = nn.LSTM(
            input_size=in_ch,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.fc = nn.Linear(hidden_size * 2, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x.permute(0, 2, 1))   # (batch, F, L) -> (batch, C, L)
        lstm_out, _ = self.lstm(x.permute(0, 2, 1))  # (batch, L, hidden*2)
        return self.fc(lstm_out[:, -1, :])  # 最后一个时间步 -> (batch, H)


def build_model(n_features: int, seq_len: int, horizon: int, params: dict) -> CNN_BiLSTM:
    """用超参数字典建模型 (lr / batch_size 是训练参数, 这里忽略)。"""
    arch = {k: v for k, v in params.items() if k not in ("lr", "batch_size")}
    return CNN_BiLSTM(n_features=n_features, seq_len=seq_len, horizon=horizon, **arch)


def trace_shapes(model: CNN_BiLSTM, x: torch.Tensor) -> None:
    """按 forward 的顺序逐层运行并打印 tensor 形状。"""
    def show(t, name):
        print(f"  {str(tuple(t.shape)):<16} {name}")

    model.eval()
    with torch.no_grad():
        show(x, "输入 (batch, L, 特征数)")
        x = x.permute(0, 2, 1)
        show(x, "permute -> (batch, 特征数, L)")
        for i, layer in enumerate(model.cnn):
            x = layer(x)
            show(x, f"cnn[{i}] {layer.__class__.__name__}")
        x = x.permute(0, 2, 1)
        show(x, "permute -> (batch, L, 通道数)")
        lstm_out, (h_n, _) = model.lstm(x)
        show(lstm_out, "BiLSTM 输出 (batch, L, hidden*2)")
        show(h_n, "  h_n (层数*2, batch, hidden)")
        out = model.fc(lstm_out[:, -1, :])
        show(out, "取最后时间步 + Linear -> (batch, H)")


if __name__ == "__main__":
    import argparse
    import json
    import sys
    from pathlib import Path

    import numpy as np

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from experiments.common import load_params, sample_dir

    parser = argparse.ArgumentParser(description="打印 CNN-BiLSTM 每层 tensor 形状")
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--batch", type=int, default=4)
    args = parser.parse_args()

    meta = json.loads((sample_dir(args.horizon) / "meta.json").read_text(encoding="utf-8"))
    params = load_params(args.horizon)
    model = build_model(meta["n_features"], meta["lookback"], args.horizon, params)
    print(f"H={args.horizon} L={meta['lookback']} 特征数={meta['n_features']} 超参数={params}")
    print(f"可训练参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n")
    print(model, "\n")
    x = torch.from_numpy(np.load(sample_dir(args.horizon) / "X_test.npy", mmap_mode="r")[: args.batch].copy())
    trace_shapes(model, x)
