"""
毛刺感知增强模型 (Surge-Aware Enhanced Models)

支持毛刺特征输入的 CNN-BiLSTM 变体：
1. SurgeCNNBiLSTM: 接受毛刺特征作为额外输入通道
2. SurgeGateCNNBiLSTM: 使用门控机制融合毛刺特征
3. SurgeAttentionCNNBiLSTM: 使用注意力机制关注毛刺时刻

Author: AI Assistant
Date: 2026-09-27
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict


class SurgeCNNBiLSTM(nn.Module):
    """
    毛刺感知 CNN-BiLSTM
    
    在标准 CNN-BiLSTM 基础上增加毛刺特征输入通道，
    毛刺特征会在 CNN 处理前与主特征拼接
    """
    
    def __init__(
        self,
        n_features: int,
        seq_len: int,
        horizon: int,
        n_surge_features: int = 3,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        cnn_channels: list[int] | tuple[int, ...] = (32, 64),
        kernel_size: int = 3,
        surge_embed_dim: int = 16,
        use_surge_gate: bool = True,
    ):
        """
        Args:
            n_features: 主特征数量（气象+时间等）
            seq_len: 输入序列长度
            horizon: 预测步长
            n_surge_features: 毛刺特征数量（方向、强度、掩码等）
            hidden_size: LSTM 隐藏层大小
            num_layers: LSTM 层数
            dropout: Dropout 比例
            cnn_channels: CNN 通道数列表
            kernel_size: 卷积核大小
            surge_embed_dim: 毛刺特征嵌入维度
            use_surge_gate: 是否使用毛刺门控机制
        """
        super().__init__()
        
        self.n_features = n_features
        self.n_surge_features = n_surge_features
        self.seq_len = seq_len
        self.horizon = horizon
        self.use_surge_gate = use_surge_gate
        
        # 毛刺特征嵌入层
        self.surge_embed = nn.Sequential(
            nn.Linear(n_surge_features, surge_embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # 总输入特征维度 = 主特征 + 毛刺嵌入
        total_features = n_features + surge_embed_dim
        
        # CNN blocks
        cnn_layers = []
        in_ch = total_features
        for out_ch in cnn_channels:
            cnn_layers.append(
                nn.Conv1d(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2
                )
            )
            cnn_layers.append(nn.BatchNorm1d(out_ch))
            cnn_layers.append(nn.ReLU())
            cnn_layers.append(nn.Dropout(dropout))
            in_ch = out_ch
        self.cnn = nn.Sequential(*cnn_layers)
        
        # 毛刺门控（可选）
        if use_surge_gate:
            self.surge_gate = nn.Sequential(
                nn.Linear(surge_embed_dim, cnn_channels[-1]),
                nn.Sigmoid(),
            )
        
        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=cnn_channels[-1] if cnn_channels else total_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        
        # 全连接层
        lstm_output_size = hidden_size * 2  # bidirectional
        self.fc = nn.Linear(lstm_output_size, horizon)
        
    def forward(
        self,
        x: torch.Tensor,
        surge_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 主输入 (batch, seq_len, n_features)
            surge_features: 毛刺特征 (batch, seq_len, n_surge_features) 或
                           (batch, n_surge_features) 样本级特征
            
        Returns:
            预测值 (batch, horizon)
        """
        batch_size, seq_len, _ = x.shape
        
        # 处理毛刺特征
        if surge_features is not None:
            if surge_features.dim() == 2:
                # 样本级特征：广播到所有时间步
                surge_features = surge_features.unsqueeze(1).expand(-1, seq_len, -1)
            
            # 嵌入毛刺特征
            surge_embedded = self.surge_embed(surge_features)  # (batch, seq_len, surge_embed_dim)
            
            # 拼接主特征和毛刺特征
            x = torch.cat([x, surge_embedded], dim=-1)  # (batch, seq_len, total_features)
        else:
            # 无毛刺特征时使用零填充
            zero_surge = torch.zeros(
                batch_size, seq_len, self.n_surge_features,
                device=x.device, dtype=x.dtype
            )
            surge_embedded = self.surge_embed(zero_surge)
            x = torch.cat([x, surge_embedded], dim=-1)
        
        # CNN 期望 (batch, channels, seq_len)
        x = x.permute(0, 2, 1)  # (batch, total_features, seq_len)
        x = self.cnn(x)  # (batch, last_cnn_ch, seq_len)
        
        # 应用毛刺门控（可选）
        if self.use_surge_gate and surge_features is not None:
            gate = self.surge_gate(surge_embedded.mean(dim=1))  # (batch, cnn_ch)
            x = x * gate.unsqueeze(-1)
        
        # LSTM 期望 (batch, seq_len, features)
        x = x.permute(0, 2, 1)  # (batch, seq_len, last_cnn_ch)
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden*2)
        
        # 取最后一个时间步的输出
        out = self.fc(lstm_out[:, -1, :])  # (batch, horizon)
        
        return out


class SurgeGateCNNBiLSTM(nn.Module):
    """
    毛刺门控增强 CNN-BiLSTM
    
    使用门控机制动态调整毛刺特征的影响
    """
    
    def __init__(
        self,
        n_features: int,
        seq_len: int,
        horizon: int,
        n_surge_features: int = 3,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        cnn_channels: list[int] | tuple[int, ...] = (32, 64),
        kernel_size: int = 3,
        gate_hidden_dim: int = 32,
    ):
        super().__init__()
        
        self.n_features = n_features
        self.n_surge_features = n_surge_features
        self.seq_len = seq_len
        self.horizon = horizon
        
        # 毛刺特征处理分支
        self.surge_encoder = nn.Sequential(
            nn.Linear(n_surge_features, gate_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden_dim, gate_hidden_dim),
            nn.ReLU(),
        )
        
        # 主特征编码器
        self.main_encoder = nn.Sequential(
            nn.Conv1d(n_features, cnn_channels[0], kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(cnn_channels[0]),
            nn.ReLU(),
        )
        
        # 门控单元
        self.gate_unit = nn.Sequential(
            nn.Linear(gate_hidden_dim + cnn_channels[0], gate_hidden_dim),
            nn.ReLU(),
            nn.Linear(gate_hidden_dim, cnn_channels[0]),
            nn.Sigmoid(),
        )
        
        # 后续 CNN
        cnn_layers = []
        in_ch = cnn_channels[0]
        for out_ch in cnn_channels[1:]:
            cnn_layers.append(
                nn.Conv1d(in_channels=in_ch, out_channels=out_ch,
                         kernel_size=kernel_size, padding=kernel_size//2)
            )
            cnn_layers.append(nn.BatchNorm1d(out_ch))
            cnn_layers.append(nn.ReLU())
            cnn_layers.append(nn.Dropout(dropout))
            in_ch = out_ch
        self.cnn_tail = nn.Sequential(*cnn_layers)
        
        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=cnn_channels[-1] if cnn_channels else cnn_channels[0],
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        
        lstm_output_size = hidden_size * 2
        self.fc = nn.Linear(lstm_output_size, horizon)
        
    def forward(
        self,
        x: torch.Tensor,
        surge_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, n_features)
            surge_features: (batch, seq_len, n_surge_features) 或 (batch, n_surge_features)
        """
        batch_size, seq_len, _ = x.shape
        
        # 处理毛刺特征
        if surge_features is not None:
            if surge_features.dim() == 2:
                surge_features = surge_features.unsqueeze(1).expand(-1, seq_len, -1)
            surge_encoded = self.surge_encoder(surge_features)  # (batch, seq_len, gate_hidden)
        else:
            surge_encoded = torch.zeros(
                batch_size, seq_len, 32, device=x.device, dtype=x.dtype
            )
        
        # 主特征 CNN
        x_cnn = x.permute(0, 2, 1)  # (batch, n_features, seq_len)
        x_cnn = self.main_encoder(x_cnn)  # (batch, cnn_ch[0], seq_len)
        
        # 门控计算
        x_transposed = x_cnn.permute(0, 2, 1)  # (batch, seq_len, cnn_ch[0])
        gate_input = torch.cat([surge_encoded, x_transposed], dim=-1)
        gate = self.gate_unit(gate_input)  # (batch, seq_len, cnn_ch[0])
        
        # 应用门控
        x_gated = x_transposed * gate
        x_cnn = x_gated.permute(0, 2, 1)  # (batch, cnn_ch[0], seq_len)
        
        # 后续 CNN
        x_cnn = self.cnn_tail(x_cnn)
        
        # LSTM
        x_lstm = x_cnn.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x_lstm)
        
        # 输出
        out = self.fc(lstm_out[:, -1, :])
        
        return out


class SurgeAttentionCNNBiLSTM(nn.Module):
    """
    毛刺注意力增强 CNN-BiLSTM
    
    使用注意力机制让模型自动关注毛刺时刻
    """
    
    def __init__(
        self,
        n_features: int,
        seq_len: int,
        horizon: int,
        n_surge_features: int = 3,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        cnn_channels: list[int] | tuple[int, ...] = (32, 64),
        kernel_size: int = 3,
        attention_dim: int = 64,
    ):
        super().__init__()
        
        self.n_features = n_features
        self.n_surge_features = n_surge_features
        self.seq_len = seq_len
        self.horizon = horizon
        
        # 毛刺特征嵌入
        self.surge_embed = nn.Sequential(
            nn.Linear(n_surge_features, attention_dim),
            nn.Tanh(),
        )
        
        # CNN blocks
        cnn_layers = []
        in_ch = n_features
        for out_ch in cnn_channels:
            cnn_layers.append(
                nn.Conv1d(in_channels=in_ch, out_channels=out_ch,
                         kernel_size=kernel_size, padding=kernel_size//2)
            )
            cnn_layers.append(nn.BatchNorm1d(out_ch))
            cnn_layers.append(nn.ReLU())
            cnn_layers.append(nn.Dropout(dropout))
            in_ch = out_ch
        self.cnn = nn.Sequential(*cnn_layers)
        
        cnn_out_dim = cnn_channels[-1] if cnn_channels else n_features
        
        # 注意力层
        self.attention_query = nn.Linear(cnn_out_dim, attention_dim)
        self.attention_key = nn.Linear(attention_dim, attention_dim)
        self.attention_value = nn.Linear(attention_dim, attention_dim)
        
        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=cnn_out_dim + attention_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        
        lstm_output_size = hidden_size * 2
        self.fc = nn.Linear(lstm_output_size, horizon)
        
    def forward(
        self,
        x: torch.Tensor,
        surge_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, n_features)
            surge_features: (batch, seq_len, n_surge_features)
        """
        batch_size, seq_len, _ = x.shape
        
        # CNN
        x_cnn = x.permute(0, 2, 1)
        x_cnn = self.cnn(x_cnn)  # (batch, cnn_out_dim, seq_len)
        x_seq = x_cnn.permute(0, 2, 1)  # (batch, seq_len, cnn_out_dim)
        
        # 毛刺注意力
        if surge_features is not None:
            surge_embed = self.surge_embed(surge_features)  # (batch, seq_len, attention_dim)
            
            # 注意力计算
            q = self.attention_query(x_seq)  # (batch, seq_len, attention_dim)
            k = self.attention_key(surge_embed)
            v = self.attention_value(surge_embed)
            
            # 简化的注意力
            attn_weights = torch.bmm(q, k.transpose(1, 2)) / (q.size(-1) ** 0.5)
            attn_weights = F.softmax(attn_weights, dim=-1)
            surge_context = torch.bmm(attn_weights, v)  # (batch, seq_len, attention_dim)
            
            # 融合
            x_fused = torch.cat([x_seq, surge_context], dim=-1)
        else:
            # 无毛刺特征时
            x_fused = torch.cat([
                x_seq,
                torch.zeros(batch_size, seq_len, 64, device=x.device, dtype=x.dtype)
            ], dim=-1)
        
        # LSTM
        lstm_out, _ = self.lstm(x_fused)
        
        # 输出
        out = self.fc(lstm_out[:, -1, :])
        
        return out


class ResidualSurgeCNNBiLSTM(nn.Module):
    """
    残差毛刺增强 CNN-BiLSTM
    
    包含残差连接和毛刺感知机制
    """
    
    def __init__(
        self,
        n_features: int,
        seq_len: int,
        horizon: int,
        n_surge_features: int = 3,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        cnn_channels: list[int] | tuple[int, ...] = (32, 64, 128),
        kernel_size: int = 3,
    ):
        super().__init__()
        
        self.n_features = n_features
        self.n_surge_features = n_surge_features
        
        # 毛刺特征处理
        self.surge_proj = nn.Linear(n_surge_features, n_features)
        
        # 残差 CNN blocks
        self.conv1 = nn.Conv1d(n_features, cnn_channels[0], kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(cnn_channels[0])
        
        self.conv2 = nn.Conv1d(cnn_channels[0], cnn_channels[1], kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(cnn_channels[1])
        
        self.conv3 = nn.Conv1d(cnn_channels[1], cnn_channels[2], kernel_size, padding=kernel_size//2)
        self.bn3 = nn.BatchNorm1d(cnn_channels[2])
        
        # 投影层（用于残差连接）
        self.proj = nn.Conv1d(n_features, cnn_channels[0], 1) if n_features != cnn_channels[0] else nn.Identity()
        
        self.dropout = nn.Dropout(dropout)
        
        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=cnn_channels[-1],
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        
        self.fc = nn.Linear(hidden_size * 2, horizon)
        
    def forward(
        self,
        x: torch.Tensor,
        surge_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, n_features)
            surge_features: (batch, seq_len, n_surge_features)
        """
        # 毛刺增强
        if surge_features is not None:
            surge_enhanced = self.surge_proj(surge_features)
            x = x + surge_enhanced
        
        # CNN
        x = x.permute(0, 2, 1)  # (batch, features, seq_len)
        
        # 残差块1
        residual = self.proj(x)  # (batch, cnn_channels[0], seq_len)
        x = self.dropout(F.relu(self.bn1(self.conv1(x))))
        # 确保形状匹配后再相加
        if x.shape[1] != residual.shape[1]:
            if x.shape[1] > residual.shape[1]:
                residual = F.pad(residual, (0, 0, 0, x.shape[1] - residual.shape[1]))
            else:
                x = F.pad(x, (0, 0, 0, residual.shape[1] - x.shape[1]))
        x = x + residual
        
        # 残差块2
        residual = x
        x = self.dropout(F.relu(self.bn2(self.conv2(x))))
        if x.shape[1] != residual.shape[1]:
            if x.shape[1] > residual.shape[1]:
                residual = F.pad(residual, (0, 0, 0, x.shape[1] - residual.shape[1]))
            else:
                x = F.pad(x, (0, 0, 0, residual.shape[1] - x.shape[1]))
        x = x + residual
        
        # 残差块3
        residual = x
        x = self.dropout(F.relu(self.bn3(self.conv3(x))))
        if x.shape[1] != residual.shape[1]:
            if x.shape[1] > residual.shape[1]:
                residual = F.pad(residual, (0, 0, 0, x.shape[1] - residual.shape[1]))
            else:
                x = F.pad(x, (0, 0, 0, residual.shape[1] - x.shape[1]))
        x = x + residual
        
        # LSTM
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        
        # 输出
        out = self.fc(lstm_out[:, -1, :])
        
        return out


def build_surge_model(
    model_name: str,
    n_features: int,
    seq_len: int,
    horizon: int,
    n_surge_features: int = 3,
    **kwargs,
) -> nn.Module:
    """
    构建毛刺感知模型
    
    Args:
        model_name: 模型名称 ('surge_cnn_bilstm', 'surge_gate_cnn_bilstm',
                    'surge_attention_cnn_bilstm', 'residual_surge_cnn_bilstm')
        n_features: 主特征数量
        seq_len: 序列长度
        horizon: 预测步长
        n_surge_features: 毛刺特征数量
        **kwargs: 其他模型参数
        
    Returns:
        模型实例
    """
    model_map = {
        'surge_cnn_bilstm': SurgeCNNBiLSTM,
        'surge_gate_cnn_bilstm': SurgeGateCNNBiLSTM,
        'surge_attention_cnn_bilstm': SurgeAttentionCNNBiLSTM,
        'residual_surge_cnn_bilstm': ResidualSurgeCNNBiLSTM,
    }
    
    if model_name not in model_map:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(model_map.keys())}")
    
    model = model_map[model_name](
        n_features=n_features,
        seq_len=seq_len,
        horizon=horizon,
        n_surge_features=n_surge_features,
        **kwargs,
    )
    
    return model


# ============================================================================
# 使用示例
# ============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("毛刺感知增强模型演示")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 测试参数
    batch_size = 8
    seq_len = 16
    horizon = 4
    n_features = 12
    n_surge_features = 3
    
    # 创建虚拟输入
    x = torch.randn(batch_size, seq_len, n_features).to(device)
    surge_features = torch.randn(batch_size, seq_len, n_surge_features).to(device)
    
    print(f"\n输入形状: x={x.shape}, surge={surge_features.shape}")
    
    # 测试各个模型
    models = [
        ('SurgeCNNBiLSTM', SurgeCNNBiLSTM(n_features, seq_len, horizon, n_surge_features)),
        ('SurgeGateCNNBiLSTM', SurgeGateCNNBiLSTM(n_features, seq_len, horizon, n_surge_features)),
        ('SurgeAttentionCNNBiLSTM', SurgeAttentionCNNBiLSTM(n_features, seq_len, horizon, n_surge_features)),
        ('ResidualSurgeCNNBiLSTM', ResidualSurgeCNNBiLSTM(n_features, seq_len, horizon, n_surge_features)),
    ]
    
    for name, model in models:
        model = model.to(device)
        model.eval()
        
        with torch.no_grad():
            out = model(x, surge_features)
            
        param_count = sum(p.numel() for p in model.parameters())
        print(f"\n{name}:")
        print(f"  输出形状: {out.shape}")
        print(f"  参数量: {param_count:,}")
    
    # 测试 build_surge_model
    print("\n" + "-" * 40)
    print("测试 build_surge_model:")
    
    test_model = build_surge_model(
        'surge_cnn_bilstm',
        n_features=n_features,
        seq_len=seq_len,
        horizon=horizon,
        n_surge_features=n_surge_features,
        hidden_size=64,
    )
    
    with torch.no_grad():
        out = test_model(x, surge_features)
        
    print(f"  模型: surge_cnn_bilstm")
    print(f"  输出形状: {out.shape}")
    
    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)
