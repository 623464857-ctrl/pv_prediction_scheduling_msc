"""
毛刺感知训练脚本 (Surge-Aware Training Pipeline)

支持毛刺特征输入和毛刺感知损失的训练流程：
1. 自动加载数据并提取毛刺特征
2. 支持多种毛刺感知模型
3. 使用毛刺感知损失函数
4. 完整的训练、验证和评估流程

Author: AI Assistant
Date: 2026-09-27
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.prediction.step3_deep_learning.exp_03_torch_utils import (
    get_device,
    predict,
    compute_metrics,
)
from experiments.prediction.step3_deep_learning.exp_03_surge_features import (
    SurgeFeatureExtractor,
    create_surge_aware_dataset,
)
from experiments.prediction.step3_deep_learning.exp_03_surge_models import (
    build_surge_model,
    SurgeCNNBiLSTM,
)
from experiments.prediction.step3_deep_learning.exp_03_surge_loss import (
    EnhancedSurgeAwareCompoundLoss,
    create_surge_labels_from_power,
    numpy_to_tensor_surge_labels,
)
from experiments.prediction.step2_hyperparameter_search.exp_02_cv_split import (
    create_rolling_folds,
)


# ============================================================================
# 配置
# ============================================================================

SURGE_MODEL_CONFIGS = {
    'surge_cnn_bilstm': {
        'hidden_size': 64,
        'num_layers': 2,
        'dropout': 0.2,
        'cnn_channels': [32, 64],
        'kernel_size': 3,
        'surge_embed_dim': 16,
        'use_surge_gate': True,
    },
    'surge_gate_cnn_bilstm': {
        'hidden_size': 64,
        'num_layers': 2,
        'dropout': 0.2,
        'cnn_channels': [32, 64],
        'kernel_size': 3,
    },
    'surge_attention_cnn_bilstm': {
        'hidden_size': 64,
        'num_layers': 2,
        'dropout': 0.2,
        'cnn_channels': [32, 64],
        'kernel_size': 3,
    },
}


# ============================================================================
# 数据处理
# ============================================================================

class SurgeDataset(torch.utils.data.Dataset):
    """
    毛刺感知数据集
    
    包含主特征和毛刺特征
    """
    
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        power_history: np.ndarray,
        surge_features: Optional[np.ndarray] = None,
        surge_threshold_ratio: float = 0.15,
    ):
        """
        Args:
            X: 主特征 (N, seq_len, n_features)
            y: 目标值 (N, horizon)
            power_history: 历史功率 (N, seq_len)
            surge_features: 预计算的毛刺特征 (N, seq_len, n_surge_features)
            surge_threshold_ratio: 毛刺检测阈值
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.n_surge_features = 3  # 方向、强度、掩码
        
        if surge_features is not None:
            self.surge_features = torch.FloatTensor(surge_features)
        else:
            # 实时计算毛刺特征
            extractor = SurgeFeatureExtractor(surge_threshold_ratio=surge_threshold_ratio)
            features = extractor.extract_surge_features(power_history)
            
            # 构建毛刺特征
            surge_dir = features['surge_direction']
            surge_mag = features['surge_magnitude']
            surge_mask = features['is_surge_sample']
            
            # 归一化强度
            surge_mag_norm = surge_mag / (surge_mag.max() + 1e-8)
            
            self.surge_features = torch.FloatTensor(
                np.stack([surge_dir, surge_mag_norm, surge_mask], axis=-1)
            )
        
        # 预计算毛刺标签
        extractor = SurgeFeatureExtractor(surge_threshold_ratio=surge_threshold_ratio)
        self.surge_labels = extractor.extract_surge_features(power_history, power_future=y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return (
            self.X[idx],
            self.y[idx],
            self.surge_features[idx],
            {k: v[idx] for k, v in self.surge_labels.items()}
        )


def create_surge_data_loader(
    X: np.ndarray,
    y: np.ndarray,
    power_history: np.ndarray,
    batch_size: int = 64,
    shuffle: bool = True,
    surge_features: Optional[np.ndarray] = None,
) -> Tuple[DataLoader, List]:
    """
    创建毛刺感知数据加载器
    """
    dataset = SurgeDataset(X, y, power_history, surge_features)
    
    def collate_fn(batch):
        X_batch, y_batch, surge_batch, labels_batch = zip(*batch)
        
        X_batch = torch.stack(X_batch)
        y_batch = torch.stack(y_batch)
        surge_batch = torch.stack(surge_batch)
        
        # 合并标签
        merged_labels = {}
        for key in labels_batch[0].keys():
            merged_labels[key] = torch.stack([l[key] for l in labels_batch])
        
        return X_batch, y_batch, surge_batch, merged_labels
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
    )
    
    return loader, dataset.surge_labels


# ============================================================================
# 训练函数
# ============================================================================

def train_surge_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_epochs: int = 50,
    patience: int = 10,
    model_name: str = 'surge_cnn_bilstm',
) -> Tuple[nn.Module, Dict]:
    """
    训练毛刺感知模型
    
    Args:
        model: 模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 设备
        max_epochs: 最大 epoch 数
        patience: 早停耐心值
        model_name: 模型名称
        
    Returns:
        (最佳模型, 训练历史)
    """
    history = {
        'train_loss': [],
        'val_loss': [],
        'loss_breakdown': [],
        'val_metrics': [],
    }
    
    best_val_loss = float('inf')
    best_model_state = None
    no_improve = 0
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    for epoch in range(max_epochs):
        epoch_start = time.time()
        
        # ----- 训练 -----
        model.train()
        train_losses = []
        train_loss_parts = []
        
        for batch_idx, (X_batch, y_batch, surge_batch, surge_labels) in enumerate(train_loader):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            surge_batch = surge_batch.to(device)
            
            # 转换标签到设备
            surge_labels = {k: v.to(device) for k, v in surge_labels.items()}
            
            optimizer.zero_grad()
            
            # 前向传播
            y_pred = model(X_batch, surge_batch)
            
            # 计算损失
            loss, loss_parts = criterion(
                y_pred, y_batch,
                power_history=X_batch[:, :, -1].unsqueeze(-1) if X_batch.size(-1) > 0 else None,
                surge_labels=surge_labels,
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
            if epoch == 0:
                train_loss_parts.append(loss_parts)
        
        avg_train_loss = np.mean(train_losses)
        
        # ----- 验证 -----
        model.eval()
        val_losses = []
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for X_batch, y_batch, surge_batch, surge_labels in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                surge_batch = surge_batch.to(device)
                surge_labels = {k: v.to(device) for k, v in surge_labels.items()}
                
                y_pred = model(X_batch, surge_batch)
                
                loss, _ = criterion(
                    y_pred, y_batch,
                    surge_labels=surge_labels,
                )
                
                val_losses.append(loss.item())
                all_preds.append(y_pred.cpu().numpy())
                all_targets.append(y_batch.cpu().numpy())
        
        avg_val_loss = np.mean(val_losses)
        
        # 计算验证指标
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        val_metrics = compute_metrics(all_targets, all_preds)
        
        # 记录历史
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_metrics'].append(val_metrics)
        
        # 合并损失分解
        if train_loss_parts:
            avg_parts = {}
            for key in train_loss_parts[0].keys():
                avg_parts[key] = np.mean([p.get(key, 0) for p in train_loss_parts])
            history['loss_breakdown'].append(avg_parts)
        
        # 学习率调整
        scheduler.step(avg_val_loss)
        
        # 早停
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        
        epoch_time = time.time() - epoch_start
        
        # 打印进度
        if (epoch + 1) % 5 == 0 or no_improve >= patience:
            print(f"Epoch {epoch+1:3d} | "
                  f"Train: {avg_train_loss:.4f} | "
                  f"Val: {avg_val_loss:.4f} | "
                  f"RMSE: {val_metrics.get('RMSE', 0):.4f} | "
                  f"Time: {epoch_time:.1f}s | "
                  f"Best: {best_val_loss:.4f}")
        
        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # 恢复最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, history


def train_with_surge_loss_only(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    loss_config: Optional[Dict] = None,
    optimizer_config: Optional[Dict] = None,
    max_epochs: int = 50,
    patience: int = 10,
    power_history_train: Optional[np.ndarray] = None,
    power_history_val: Optional[np.ndarray] = None,
) -> Tuple[nn.Module, Dict]:
    """
    使用毛刺感知损失训练模型（不需要毛刺输入）
    
    适用于标准模型 + 毛刺感知损失的场景
    """
    if loss_config is None:
        loss_config = {
            'mse_weight': 1.0,
            'surge_weight': 1.5,
            'direction_weight': 0.5,
            'intensity_weight': 0.5,
            'use_ramp': True,
            'use_volatility': True,
            'use_physics': True,
        }
    
    if optimizer_config is None:
        optimizer_config = {
            'lr': 0.001,
            'weight_decay': 1e-5,
        }
    
    # 创建损失函数
    criterion = EnhancedSurgeAwareCompoundLoss(**loss_config).to(device)
    
    # 创建优化器
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=optimizer_config['lr'],
        weight_decay=optimizer_config.get('weight_decay', 0),
    )
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # 训练历史
    history = {
        'train_loss': [],
        'val_loss': [],
        'loss_breakdown': [],
        'val_metrics': [],
    }
    
    best_val_loss = float('inf')
    best_model_state = None
    no_improve = 0
    
    # 创建毛刺标签计算器
    from experiments.prediction.step3_deep_learning.exp_03_surge_features import (
        SurgeFeatureExtractor
    )
    extractor = SurgeFeatureExtractor()
    
    for epoch in range(max_epochs):
        # ----- 训练 -----
        model.train()
        train_losses = []
        train_loss_parts = []
        
        batch_idx = 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            # 获取对应的历史功率
            if power_history_train is not None:
                start_idx = batch_idx * train_loader.batch_size
                end_idx = start_idx + len(X_batch)
                hist_idx = np.arange(start_idx, min(end_idx, len(power_history_train)))
                hist_idx = hist_idx[hist_idx < len(power_history_train)]
                
                if len(hist_idx) > 0:
                    power_hist = power_history_train[hist_idx]
                    # 创建毛刺标签
                    surge_labels = extractor.extract_surge_features(power_hist, power_future=y_batch.cpu().numpy())
                    surge_labels = numpy_to_tensor_surge_labels(surge_labels, device)
                else:
                    surge_labels = None
            else:
                surge_labels = None
            
            optimizer.zero_grad()
            y_pred = model(X_batch)
            
            # 处理输出形状
            if y_pred.dim() == 3:
                y_pred = y_pred[:, -1, :]
            
            # 计算损失
            loss, loss_parts = criterion(
                y_pred, y_batch,
                power_history=X_batch[:, :, -1].unsqueeze(-1) if X_batch.size(-1) > 0 else None,
                surge_labels=surge_labels,
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
            if epoch == 0:
                train_loss_parts.append(loss_parts)
            
            batch_idx += 1
        
        avg_train_loss = np.mean(train_losses)
        
        # ----- 验证 -----
        model.eval()
        val_losses = []
        all_preds = []
        all_targets = []
        
        batch_idx = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                
                if power_history_val is not None:
                    start_idx = batch_idx * val_loader.batch_size
                    end_idx = start_idx + len(X_batch)
                    hist_idx = np.arange(start_idx, min(end_idx, len(power_history_val)))
                    hist_idx = hist_idx[hist_idx < len(power_history_val)]
                    
                    if len(hist_idx) > 0:
                        power_hist = power_history_val[hist_idx]
                        surge_labels = extractor.extract_surge_features(power_hist, power_future=y_batch.cpu().numpy())
                        surge_labels = numpy_to_tensor_surge_labels(surge_labels, device)
                    else:
                        surge_labels = None
                else:
                    surge_labels = None
                
                y_pred = model(X_batch)
                if y_pred.dim() == 3:
                    y_pred = y_pred[:, -1, :]
                
                loss, _ = criterion(
                    y_pred, y_batch,
                    surge_labels=surge_labels,
                )
                
                val_losses.append(loss.item())
                all_preds.append(y_pred.cpu().numpy())
                all_targets.append(y_batch.cpu().numpy())
                
                batch_idx += 1
        
        avg_val_loss = np.mean(val_losses)
        
        # 计算验证指标
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        val_metrics = compute_metrics(all_targets, all_preds)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_metrics'].append(val_metrics)
        
        if train_loss_parts:
            avg_parts = {}
            for key in train_loss_parts[0].keys():
                avg_parts[key] = np.mean([p.get(key, 0) for p in train_loss_parts])
            history['loss_breakdown'].append(avg_parts)
        
        scheduler.step(avg_val_loss)
        
        # 早停
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        
        if (epoch + 1) % 5 == 0 or no_improve >= patience:
            print(f"Epoch {epoch+1:3d} | "
                  f"Train: {avg_train_loss:.4f} | "
                  f"Val: {avg_val_loss:.4f} | "
                  f"RMSE: {val_metrics.get('RMSE', 0):.4f}")
        
        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, history


# ============================================================================
# 主函数
# ============================================================================

def run_surge_training(
    data_path: str,
    horizon: int,
    model_type: str = 'surge_cnn_bilstm',
    surge_threshold_ratio: float = 0.15,
    batch_size: int = 64,
    lr: float = 0.001,
    max_epochs: int = 50,
    patience: int = 10,
    n_folds: int = 3,
    seed: int = 42,
    use_surge_features: bool = True,
    device: Optional[torch.device] = None,
) -> Dict:
    """
    运行毛刺感知训练
    
    Args:
        data_path: 数据路径（应包含 X_*.npy, y_*.npy）
        horizon: 预测步长
        model_type: 模型类型
        surge_threshold_ratio: 毛刺检测阈值
        batch_size: 批次大小
        lr: 学习率
        max_epochs: 最大 epoch 数
        patience: 早停耐心值
        n_folds: 交叉验证折数
        seed: 随机种子
        use_surge_features: 是否使用毛刺特征输入
        device: 设备
        
    Returns:
        训练结果字典
    """
    if device is None:
        device = get_device()
    
    print("=" * 60)
    print(f"毛刺感知训练 - horizon={horizon}, model={model_type}")
    print("=" * 60)
    
    # 加载数据
    data_path = Path(data_path)
    X_train = np.load(data_path / "X_train_seq.npy")
    y_train = np.load(data_path / "y_train.npy")
    X_val = np.load(data_path / "X_val_seq.npy")
    y_val = np.load(data_path / "y_val.npy")
    X_test = np.load(data_path / "X_test_seq.npy")
    y_test = np.load(data_path / "y_test.npy")
    
    n_samples, seq_len, n_features = X_train.shape
    
    print(f"\n数据形状:")
    print(f"  训练: X={X_train.shape}, y={y_train.shape}")
    print(f"  验证: X={X_val.shape}, y={y_val.shape}")
    print(f"  测试: X={X_test.shape}, y={y_test.shape}")
    
    # 提取历史功率（假设最后一列是功率）
    power_col_idx = -1  # 可根据实际情况调整
    power_history_train = X_train[:, :, power_col_idx]
    power_history_val = X_val[:, :, power_col_idx]
    power_history_test = X_test[:, :, power_col_idx]
    
    # 创建数据加载器
    if use_surge_features:
        train_loader, surge_labels = create_surge_data_loader(
            X_train, y_train, power_history_train,
            batch_size=batch_size, shuffle=True,
        )
        val_loader, _ = create_surge_data_loader(
            X_val, y_val, power_history_val,
            batch_size=batch_size, shuffle=False,
        )
        test_loader, _ = create_surge_data_loader(
            X_test, y_test, power_history_test,
            batch_size=batch_size, shuffle=False,
        )
    else:
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train),
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.FloatTensor(y_val),
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = None
    
    # 获取模型配置
    model_config = SURGE_MODEL_CONFIGS.get(model_type, {})
    
    # 设置随机种子
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # 构建模型
    if use_surge_features:
        model = build_surge_model(
            model_type,
            n_features=n_features,
            seq_len=seq_len,
            horizon=horizon,
            n_surge_features=3,
            **model_config,
        )
    else:
        from experiments.prediction.step3_deep_learning.exp_03_models import build_model
        model = build_model(
            'cnn_bilstm',
            n_features=n_features,
            seq_len=seq_len,
            horizon=horizon,
            **model_config,
        )
    
    model = model.to(device)
    
    # 创建损失函数
    if use_surge_features:
        criterion = EnhancedSurgeAwareCompoundLoss(
            mse_weight=1.0,
            surge_weight=1.5,
            direction_weight=0.5,
            intensity_weight=0.5,
            use_ramp=True,
            use_volatility=True,
            use_physics=True,
        ).to(device)
    else:
        criterion = EnhancedSurgeAwareCompoundLoss(
            mse_weight=1.0,
            surge_weight=1.0,
            direction_weight=0.3,
            intensity_weight=0.3,
            use_ramp=True,
            use_physics=True,
        ).to(device)
    
    # 创建优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    print(f"\n模型: {model_type}")
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"设备: {device}")
    
    # 训练
    print("\n开始训练...")
    
    if use_surge_features:
        model, history = train_surge_model(
            model, train_loader, val_loader,
            criterion, optimizer, device,
            max_epochs=max_epochs,
            patience=patience,
            model_name=model_type,
        )
    else:
        model, history = train_with_surge_loss_only(
            model, train_loader, val_loader,
            device,
            max_epochs=max_epochs,
            patience=patience,
            power_history_train=power_history_train,
            power_history_val=power_history_val,
        )
    
    # 在测试集上评估
    print("\n测试集评估...")
    model.eval()
    
    if use_surge_features:
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for X_batch, y_batch, surge_batch, _ in test_loader:
                X_batch = X_batch.to(device)
                surge_batch = surge_batch.to(device)
                y_pred = model(X_batch, surge_batch)
                all_preds.append(y_pred.cpu().numpy())
                all_targets.append(y_batch.numpy())
    else:
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                y_pred = model(X_batch)
                if y_pred.dim() == 3:
                    y_pred = y_pred[:, -1, :]
                all_preds.append(y_pred.cpu().numpy())
                all_targets.append(y_batch.numpy())
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    test_metrics = compute_metrics(all_targets, all_preds)
    
    print("\n" + "=" * 60)
    print("最终结果")
    print("=" * 60)
    print(f"最佳验证损失: {min(history['val_loss']):.4f}")
    print(f"测试集指标:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")
    
    # 保存结果
    results = {
        'model_type': model_type,
        'horizon': horizon,
        'use_surge_features': use_surge_features,
        'best_val_loss': min(history['val_loss']),
        'test_metrics': test_metrics,
        'history': history,
    }
    
    return results


# ============================================================================
# 命令行入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="毛刺感知训练")
    
    # 数据参数
    parser.add_argument("--data-path", type=str, required=True,
                        help="样本数据目录路径")
    parser.add_argument("--horizon", type=int, required=True,
                        help="预测步长 (1, 4, 16)")
    
    # 模型参数
    parser.add_argument("--model", type=str, default="surge_cnn_bilstm",
                        choices=['surge_cnn_bilstm', 'surge_gate_cnn_bilstm',
                                'surge_attention_cnn_bilstm', 'cnn_bilstm'],
                        help="模型类型")
    parser.add_argument("--use-surge-features", action="store_true",
                        help="使用毛刺特征输入")
    
    # 训练参数
    parser.add_argument("--batch-size", type=int, default=64,
                        help="批次大小")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="学习率")
    parser.add_argument("--epochs", type=int, default=50,
                        help="最大 epoch 数")
    parser.add_argument("--patience", type=int, default=10,
                        help="早停耐心值")
    
    # 其他参数
    parser.add_argument("--surge-threshold", type=float, default=0.15,
                        help="毛刺检测阈值")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    
    args = parser.parse_args()
    
    results = run_surge_training(
        data_path=args.data_path,
        horizon=args.horizon,
        model_type=args.model,
        surge_threshold_ratio=args.surge_threshold,
        batch_size=args.batch_size,
        lr=args.lr,
        max_epochs=args.epochs,
        patience=args.patience,
        seed=args.seed,
        use_surge_features=args.use_surge_features,
    )
    
    return results


if __name__ == "__main__":
    main()
