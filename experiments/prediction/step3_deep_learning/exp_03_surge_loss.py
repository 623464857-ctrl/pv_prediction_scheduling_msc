"""
毛刺感知损失函数 (Surge-Aware Loss Functions)

针对发电突变预测的专用损失函数设计，包含：
1. 波动感知重加权 (Volatility-Aware Re-weighting)
2. 爬坡加权损失 (Ramp-Weighted Loss)
3. 物理引导的复合目标 (Physics-Guided Compound Loss)
4. Focal Loss for Volatility
5. 边缘感知损失 (Edge-Aware Loss)
6. 增强版毛刺感知复合损失 (EnhancedSurgeAwareCompoundLoss)

Author: AI Assistant
Date: 2026-09-27
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from torch.utils.data import DataLoader, WeightedRandomSampler
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# 第一部分：样本权重计算器
# ============================================================================

class VolatilityCalculator:
    """波动率计算器 - 用于计算样本权重"""
    
    def __init__(
        self,
        volatility_threshold_percentile: float = 75.0,
        ramp_threshold_percentile: float = 75.0,
    ):
        """
        Args:
            volatility_threshold_percentile: 波动率阈值百分位（用于标记高波动样本）
            ramp_threshold_percentile: 爬坡阈值百分位
        """
        self.volatility_percentile = volatility_threshold_percentile
        self.ramp_percentile = ramp_threshold_percentile
        
    def calculate_power_volatility(self, power_history: np.ndarray) -> np.ndarray:
        """
        计算发电功率波动率
        
        Args:
            power_history: 历史发电功率序列 (N, seq_len)
            
        Returns:
            波动率数组 (N,)
        """
        if power_history.ndim == 1:
            power_history = power_history.reshape(-1, 1)
            
        # 方法1：一阶差分绝对值（爬坡幅度）
        first_diff = np.abs(np.diff(power_history, axis=1))
        ramp_magnitude = np.mean(first_diff, axis=1)
        
        # 方法2：标准差（整体波动）
        power_std = np.std(power_history, axis=1)
        
        # 方法3：峰度（尖峰程度）
        power_mean = np.mean(power_history, axis=1, keepdims=True)
        power_normalized = power_history - power_mean
        kurtosis = np.mean(power_normalized ** 4, axis=1) / (np.var(power_history, axis=1) ** 2 + 1e-8) - 3
        
        # 综合波动率
        volatility = 0.5 * ramp_magnitude + 0.3 * power_std + 0.2 * np.abs(kurtosis)
        
        return volatility
    
    def calculate_sample_weights(
        self,
        power_history: np.ndarray,
        target_power: np.ndarray,
        base_weight: float = 1.0,
        alpha: float = 2.0,
    ) -> np.ndarray:
        """
        计算样本权重
        
        权重公式: weight = base_weight * (1 + alpha * normalized_volatility)
        
        Args:
            power_history: 历史发电功率 (N, seq_len)
            target_power: 目标发电功率 (N,) 或 (N, horizon)
            base_weight: 基础权重
            alpha: 波动放大系数
            
        Returns:
            样本权重 (N,)
        """
        volatility = self.calculate_power_volatility(power_history)
        
        # 归一化波动率到 [0, 1]
        if volatility.max() > volatility.min():
            volatility_normalized = (volatility - volatility.min()) / (volatility.max() - volatility.min() + 1e-8)
        else:
            volatility_normalized = np.zeros_like(volatility)
        
        # 计算权重
        weights = base_weight * (1.0 + alpha * volatility_normalized)
        
        return weights
    
    def calculate_ramp_weights(
        self,
        power_history: np.ndarray,
        ramp_weight_up: float = 2.0,
        ramp_weight_down: float = 2.0,
    ) -> np.ndarray:
        """
        计算爬坡方向权重（骤升/骤降）
        
        Args:
            power_history: 历史发电功率 (N, seq_len)
            ramp_weight_up: 骤升时权重倍数
            ramp_weight_down: 骤降时权重倍数
            
        Returns:
            爬坡权重 (N,)
        """
        # 计算最后两个时间步的变化
        if power_history.ndim == 1:
            power_diff = power_history[-1] - power_history[-2]
        else:
            power_diff = power_history[:, -1] - power_history[:, -2]
        
        weights = np.ones_like(power_diff)
        weights[power_diff > 0] = ramp_weight_up   # 骤升
        weights[power_diff < 0] = ramp_weight_down  # 骤降
        
        return weights


# ============================================================================
# 第二部分：损失函数类（基础版）
# ============================================================================

class VolatilityWeightedLoss(nn.Module):
    """
    波动加权损失
    
    对高波动样本给予更高权重
    
    loss = weight * MSE(y_pred, y_true)
    weight = 1 + alpha * volatility_normalized
    """
    
    def __init__(self, alpha: float = 2.0, reduction: str = 'mean'):
        """
        Args:
            alpha: 波动放大系数，越大高波动样本权重越高
            reduction: 损失聚合方式 ('mean', 'sum', 'none')
        """
        super().__init__()
        self.alpha = alpha
        self.reduction = reduction
        
    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        volatility: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            y_pred: 预测值 (batch, ...) 或 (batch, horizon)
            y_true: 真实值 (batch, ...) 或 (batch, horizon)
            volatility: 波动率 (batch,)，如果为None则使用梯度作为波动代理
            weights: 预计算的权重 (batch,)
            
        Returns:
            加权损失
        """
        # 计算基础MSE
        mse = F.mse_loss(y_pred, y_true, reduction='none')
        
        # 展平以便于加权
        if mse.dim() > 1:
            mse = mse.mean(dim=-1)  # (batch,)
        
        # 计算或使用提供的权重
        if weights is not None:
            sample_weights = weights
        elif volatility is not None:
            # 从波动率计算权重
            vol_norm = (volatility - volatility.min()) / (volatility.max() - volatility.min() + 1e-8)
            sample_weights = 1.0 + self.alpha * vol_norm
        else:
            # 使用梯度作为波动代理
            if y_pred.grad is not None:
                gradient_magnitude = torch.abs(y_pred.grad).mean(dim=-1)
                vol_norm = gradient_magnitude / (gradient_magnitude.max() + 1e-8)
                sample_weights = 1.0 + self.alpha * vol_norm
            else:
                sample_weights = torch.ones_like(mse)
        
        # 应用权重
        weighted_mse = mse * sample_weights
        
        # 聚合
        if self.reduction == 'mean':
            return weighted_mse.mean()
        elif self.reduction == 'sum':
            return weighted_mse.sum()
        else:
            return weighted_mse


class RampWeightedLoss(nn.Module):
    """
    爬坡加权损失 (Ramp-Weighted Loss)
    
    专门针对功率骤变时刻的加权损失
    公式: weight = base + ramp_bonus * is_ramp
    """
    
    def __init__(
        self,
        ramp_threshold: float = 0.1,
        ramp_weight: float = 3.0,
        ramp_type: str = 'both',  # 'up', 'down', 'both'
        reduction: str = 'mean',
    ):
        """
        Args:
            ramp_threshold: 爬坡阈值（相对于最大功率的比例）
            ramp_weight: 爬坡样本权重倍数
            ramp_type: 爬坡类型 ('up'=骤升, 'down'=骤降, 'both'=两者)
            reduction: 损失聚合方式
        """
        super().__init__()
        self.ramp_threshold = ramp_threshold
        self.ramp_weight = ramp_weight
        self.ramp_type = ramp_type
        self.reduction = reduction
        
    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        power_history: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            y_pred: 预测值 (batch, horizon) 或 (batch,)
            y_true: 真实值 (batch, horizon) 或 (batch,)
            power_history: 历史功率 (batch, seq_len)，用于检测爬坡
            
        Returns:
            爬坡加权损失
        """
        # 计算基础MSE
        mse = F.mse_loss(y_pred, y_true, reduction='none')
        if mse.dim() > 1:
            mse = mse.mean(dim=-1)
        
        # 计算权重
        if power_history is not None:
            # 处理 (batch, seq_len, 1) 或 (batch, seq_len) 形状
            if power_history.dim() == 3:
                power_history = power_history.squeeze(-1)  # (batch, seq_len)
            
            # 从历史功率检测爬坡
            power_diff = power_history[:, -1] - power_history[:, -2]  # (batch,)
            power_max = power_history.max(dim=1)[0]  # (batch,)
            
            # 相对变化率
            relative_change = power_diff / (power_max + 1e-8)  # (batch,)
            
            # 检测爬坡类型
            is_ramp_up = relative_change > self.ramp_threshold
            is_ramp_down = relative_change < -self.ramp_threshold
            
            # 根据类型设置权重
            weights = torch.ones_like(mse)  # (batch,)
            
            if self.ramp_type in ['up', 'both']:
                weights[is_ramp_up] = self.ramp_weight
            if self.ramp_type in ['down', 'both']:
                weights[is_ramp_down] = self.ramp_weight
        else:
            # 使用预测误差本身作为爬坡信号
            error = torch.abs(y_pred - y_true)
            error_std = error.std() + 1e-8
            is_ramp = error > 2 * error_std
            weights = torch.ones_like(mse)
            weights[is_ramp] = self.ramp_weight
        
        weighted_mse = mse * weights
        
        if self.reduction == 'mean':
            return weighted_mse.mean()
        elif self.reduction == 'sum':
            return weighted_mse.sum()
        else:
            return weighted_mse


class PhysicsGuidedCompoundLoss(nn.Module):
    """
    物理引导的复合损失
    
    结合多个物理约束的损失项：
    1. MSE主损失
    2. 梯度平滑项（避免剧烈波动）
    3. 边界惩罚项（惩罚负值预测）
    4. 爬坡感知项
    """
    
    def __init__(
        self,
        lambda_smooth: float = 0.1,      # 平滑项权重
        lambda_boundary: float = 0.2,     # 边界惩罚权重
        lambda_ramp: float = 0.3,        # 爬坡感知权重
        ramp_weight: float = 2.0,
        min_power: float = 0.0,
        max_power: float = None,
    ):
        super().__init__()
        self.lambda_smooth = lambda_smooth
        self.lambda_boundary = lambda_boundary
        self.lambda_ramp = lambda_ramp
        self.ramp_weight = ramp_weight
        self.min_power = min_power
        self.max_power = max_power
        
    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        power_history: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            y_pred: 预测值
            y_true: 真实值
            power_history: 历史功率 (batch, seq_len)
            
        Returns:
            (总损失, 损失分解字典)
        """
        losses = {}
        
        # 1. MSE主损失
        mse_loss = F.mse_loss(y_pred, y_true)
        losses['mse'] = mse_loss.item()
        
        # 2. 梯度平滑项（预测的相邻时间步应该平滑）
        if y_pred.dim() == 2 and y_pred.size(1) > 1:
            pred_diff = y_pred[:, 1:] - y_pred[:, :-1]
            true_diff = y_true[:, 1:] - y_true[:, :-1]
            smooth_loss = F.mse_loss(pred_diff, true_diff)
        else:
            smooth_loss = torch.tensor(0.0, device=y_pred.device)
        losses['smooth'] = smooth_loss.item()
        
        # 3. 边界惩罚项
        boundary_loss = torch.tensor(0.0, device=y_pred.device)
        if self.min_power is not None:
            # 惩罚预测低于最小值
            below_min = F.relu(self.min_power - y_pred)
            boundary_loss = boundary_loss + below_min.mean()
        if self.max_power is not None:
            # 惩罚预测高于最大值
            above_max = F.relu(y_pred - self.max_power)
            boundary_loss = boundary_loss + above_max.mean()
        losses['boundary'] = boundary_loss.item()
        
        # 4. 爬坡感知项
        if power_history is not None:
            # 计算历史爬坡
            history_diff = power_history[:, -1] - power_history[:, -2]
            # 预测也应该遵循类似的爬坡方向
            pred_first_step = y_pred[:, 0] if y_pred.dim() > 1 else y_pred
            
            # 同向奖励，异向惩罚
            ramp_direction = torch.sign(history_diff)
            pred_direction = torch.sign(pred_first_step - power_history[:, -1])
            
            # 一致性损失
            ramp_loss = F.relu(-ramp_direction * (pred_first_step - power_history[:, -1]) + 1.0).mean()
            
            # 高爬坡时刻额外加权
            ramp_magnitude = torch.abs(history_diff)
            ramp_threshold = ramp_magnitude.mean() + ramp_magnitude.std()
            high_ramp_mask = ramp_magnitude > ramp_threshold
            
            if high_ramp_mask.any():
                ramp_loss = ramp_loss * self.ramp_weight
        else:
            ramp_loss = torch.tensor(0.0, device=y_pred.device)
        losses['ramp'] = ramp_loss.item()
        
        # 总损失
        total_loss = (
            mse_loss +
            self.lambda_smooth * smooth_loss +
            self.lambda_boundary * boundary_loss +
            self.lambda_ramp * ramp_loss
        )
        
        return total_loss, losses


class FocalLossForVolatility(nn.Module):
    """
    Focal Loss for Volatility
    
    借鉴目标检测中的Focal Loss思想，降低易分类样本的权重，
    聚焦于难分类的高波动样本
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean',
    ):
        """
        Args:
            alpha: 类别权重
            gamma: 聚焦参数，gamma越大越聚焦于难分类样本
            reduction: 损失聚合方式
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        volatility: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            y_pred: 预测值
            y_true: 真实值
            volatility: 波动率 (batch,)，用于调整alpha
        """
        # 计算MSE
        mse = F.mse_loss(y_pred, y_true, reduction='none')
        if mse.dim() > 1:
            mse = mse.mean(dim=-1)
        
        # 归一化MSE到 [0, 1] 作为"难度"指标
        mse_normalized = mse / (mse.max() + 1e-8)
        
        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1 - mse_normalized) ** self.gamma
        
        # Alpha weight: 可根据波动率调整
        if volatility is not None:
            vol_norm = volatility / (volatility.max() + 1e-8)
            alpha_t = self.alpha + (1 - self.alpha) * vol_norm
        else:
            alpha_t = self.alpha
        
        # Focal Loss: alpha * (1-p)^gamma * log(p)
        # 这里我们用 (1-mse) 作为概率近似
        prob = 1 - mse_normalized
        focal_loss = alpha_t * focal_weight * (-torch.log(prob + 1e-8))
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class EdgeAwareLoss(nn.Module):
    """
    边缘感知损失
    
    对发电曲线的边缘（突变点）给予特别关注
    使用梯度的二范数作为边缘检测代理
    """
    
    def __init__(
        self,
        edge_weight: float = 3.0,
        edge_threshold: float = 0.1,
        reduction: str = 'mean',
    ):
        super().__init__()
        self.edge_weight = edge_weight
        self.edge_threshold = edge_threshold
        self.reduction = reduction
        
    def compute_gradient(self, x: torch.Tensor) -> torch.Tensor:
        """计算一阶差分（梯度）"""
        return x[:, 1:] - x[:, :-1]
    
    def compute_second_gradient(self, x: torch.Tensor) -> torch.Tensor:
        """计算二阶差分（曲率/边缘）"""
        grad = self.compute_gradient(x)
        return grad[:, 1:] - grad[:, :-1]
    
    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        power_history: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            y_pred: 预测值 (batch, horizon) 或 (batch,)
            y_true: 真实值 (batch, horizon) 或 (batch,)
            power_history: 历史功率 (batch, seq_len)
        """
        # 主损失MSE
        mse = F.mse_loss(y_pred, y_true, reduction='none')
        if mse.dim() > 1:
            mse = mse.mean(dim=-1)
        
        # 计算边缘感知权重
        if power_history is not None:
            # 从历史数据检测边缘点
            history_gradient = self.compute_gradient(power_history)
            history_second_grad = self.compute_second_gradient(power_history)
            
            # 边缘强度 = |二阶导数| 或 |一阶导数变化|
            edge_strength = torch.abs(history_second_grad).mean(dim=1)
            
            # 归一化
            edge_norm = edge_strength / (edge_strength.max() + 1e-8)
            
            # 边缘权重
            edge_weights = 1.0 + self.edge_weight * edge_norm
        else:
            # 从预测误差推断边缘
            error = torch.abs(y_pred - y_true)
            error_norm = error / (error.max() + 1e-8)
            edge_weights = 1.0 + self.edge_weight * error_norm
        
        weighted_mse = mse * edge_weights
        
        if self.reduction == 'mean':
            return weighted_mse.mean()
        elif self.reduction == 'sum':
            return weighted_mse.sum()
        else:
            return weighted_mse


# ============================================================================
# 第三部分：基础综合损失函数
# ============================================================================

class SurgeAwareCompoundLoss(nn.Module):
    """
    综合毛刺感知损失
    
    整合多种损失策略的基础方案
    """
    
    def __init__(
        self,
        # 主损失权重
        mse_weight: float = 1.0,
        
        # 波动加权
        use_volatility: bool = True,
        volatility_alpha: float = 2.0,
        
        # 爬坡加权
        use_ramp: bool = True,
        ramp_weight: float = 2.5,
        ramp_threshold: float = 0.1,
        
        # 物理约束
        use_physics: bool = True,
        lambda_smooth: float = 0.1,
        lambda_boundary: float = 0.2,
        
        # 边缘感知
        use_edge: bool = True,
        edge_weight: float = 2.0,
        
        # 归一化
        normalize_weights: bool = True,
    ):
        super().__init__()
        
        self.use_volatility = use_volatility
        self.use_ramp = use_ramp
        self.use_physics = use_physics
        self.use_edge = use_edge
        self.normalize_weights = normalize_weights
        
        # 损失函数实例
        self.volatility_loss = VolatilityWeightedLoss(alpha=volatility_alpha)
        self.ramp_loss = RampWeightedLoss(
            ramp_threshold=ramp_threshold,
            ramp_weight=ramp_weight,
        )
        self.physics_loss = PhysicsGuidedCompoundLoss(
            lambda_smooth=lambda_smooth,
            lambda_boundary=lambda_boundary,
            lambda_ramp=ramp_weight,
        )
        self.edge_loss = EdgeAwareLoss(edge_weight=edge_weight)
        
        # 权重
        self.mse_weight = mse_weight
        self.volatility_weight = 0.3 if use_volatility else 0.0
        self.ramp_weight = 0.4 if use_ramp else 0.0
        self.physics_weight = 0.2 if use_physics else 0.0
        self.edge_weight_loss = 0.1 if use_edge else 0.0
        
    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        power_history: Optional[torch.Tensor] = None,
        volatility: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            y_pred: 预测值
            y_true: 真实值
            power_history: 历史功率 (batch, seq_len)
            volatility: 预计算的波动率 (batch,)
            
        Returns:
            (总损失, 损失分解字典)
        """
        total_loss = torch.tensor(0.0, device=y_pred.device)
        losses = {}
        
        # 1. MSE主损失
        mse = F.mse_loss(y_pred, y_true)
        total_loss = total_loss + self.mse_weight * mse
        losses['mse'] = mse.item()
        
        # 2. 波动加权损失
        if self.use_volatility:
            vol_loss = self.volatility_loss(y_pred, y_true, volatility=volatility)
            total_loss = total_loss + self.volatility_weight * vol_loss
            losses['volatility'] = vol_loss.item()
        
        # 3. 爬坡加权损失
        if self.use_ramp and power_history is not None:
            ramp_loss_val = self.ramp_loss(y_pred, y_true, power_history=power_history)
            total_loss = total_loss + self.ramp_weight * ramp_loss_val
            losses['ramp'] = ramp_loss_val.item()
        
        # 4. 物理引导损失
        if self.use_physics and power_history is not None:
            physics_loss, physics_parts = self.physics_loss(
                y_pred, y_true, power_history=power_history
            )
            total_loss = total_loss + self.physics_weight * physics_loss
            losses.update({f'physics_{k}': v for k, v in physics_parts.items()})
        
        # 5. 边缘感知损失
        if self.use_edge and power_history is not None:
            edge_loss_val = self.edge_loss(y_pred, y_true, power_history=power_history)
            total_loss = total_loss + self.edge_weight_loss * edge_loss_val
            losses['edge'] = edge_loss_val.item()
        
        return total_loss, losses


# ============================================================================
# 第四部分：增强版损失函数（毛刺感知MSE）
# ============================================================================

class SurgeAwareMSE(nn.Module):
    """
    毛刺感知 MSE 损失
    
    结合毛刺特征计算加权 MSE
    """
    
    def __init__(
        self,
        surge_weight: float = 3.0,
        ramp_weight: float = 2.0,
        reduction: str = 'mean',
    ):
        super().__init__()
        self.surge_weight = surge_weight
        self.ramp_weight = ramp_weight
        self.reduction = reduction
        
    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        surge_labels: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Args:
            y_pred: 预测值 (batch, horizon) 或 (batch,)
            y_true: 真实值 (batch, horizon) 或 (batch,)
            surge_labels: 毛刺标签字典，包含:
                - is_surge: 是否为毛刺样本 (batch,)
                - is_ramp: 是否为爬坡样本 (batch,)
                - surge_direction: 毛刺方向 (batch,)
                - max_surge_intensity: 最大毛刺强度 (batch,)
                
        Returns:
            加权 MSE 损失
        """
        # 计算基础 MSE
        mse = F.mse_loss(y_pred, y_true, reduction='none')
        if mse.dim() > 1:
            mse = mse.mean(dim=-1)  # (batch,)
        
        # 默认权重
        weights = torch.ones_like(mse)
        
        # 应用毛刺权重
        if surge_labels is not None:
            if 'is_surge' in surge_labels and surge_labels['is_surge'] is not None:
                is_surge = surge_labels['is_surge'].to(mse.device)
                weights = weights + (self.surge_weight - 1.0) * is_surge
                
            if 'is_ramp' in surge_labels and surge_labels['is_ramp'] is not None:
                is_ramp = surge_labels['is_ramp'].to(mse.device)
                weights = weights + (self.ramp_weight - 1.0) * is_ramp
                
            # 基于毛刺强度进一步加权
            if 'max_surge_intensity' in surge_labels and surge_labels['max_surge_intensity'] is not None:
                intensity = surge_labels['max_surge_intensity'].to(mse.device)
                intensity_norm = intensity / (intensity.max() + 1e-8)
                weights = weights * (1.0 + 0.5 * intensity_norm)
        
        # 应用权重
        weighted_mse = mse * weights
        
        if self.reduction == 'mean':
            return weighted_mse.mean()
        elif self.reduction == 'sum':
            return weighted_mse.sum()
        else:
            return weighted_mse


class DirectionAwareLoss(nn.Module):
    """
    方向感知损失
    
    专门惩罚方向预测错误的损失
    """
    
    def __init__(
        self,
        direction_weight: float = 2.0,
        reduction: str = 'mean',
    ):
        super().__init__()
        self.direction_weight = direction_weight
        self.reduction = reduction
        
    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        power_history: Optional[torch.Tensor] = None,
        surge_labels: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Args:
            y_pred: 预测值 (batch, horizon)
            y_true: 真实值 (batch, horizon)
            power_history: 历史功率 (batch, seq_len)
            surge_labels: 毛刺标签
        """
        # 基础 MSE
        mse = F.mse_loss(y_pred, y_true, reduction='none')
        if mse.dim() > 1:
            mse = mse.mean(dim=-1)
        
        # 方向损失
        direction_loss = torch.tensor(0.0, device=mse.device)
        
        if power_history is not None:
            # 计算预测方向和真实方向
            last_power = power_history[:, -1]  # (batch,)
            pred_change = y_pred[:, 0] - last_power  # 预测变化
            true_change = y_true[:, 0] - last_power  # 真实变化
            
            # 方向一致性损失
            pred_dir = torch.sign(pred_change)
            true_dir = torch.sign(true_change)
            
            # 方向错误惩罚
            direction_error = F.relu(-pred_dir * true_dir)  # 方向相反时为1
            direction_loss = direction_error.mean() * self.direction_weight
            
        elif surge_labels is not None and 'surge_direction' in surge_labels:
            # 使用预计算的毛刺方向
            true_dir = surge_labels['surge_direction'].to(mse.device)
            
            # 计算预测方向
            pred_dir = torch.sign(y_pred[:, 0])
            
            # 方向错误惩罚
            direction_error = F.relu(-pred_dir * true_dir)
            direction_loss = direction_error.mean() * self.direction_weight
        
        # 总损失
        total_loss = mse.mean() + direction_loss
        
        return total_loss


class SurgeIntensityWeightedLoss(nn.Module):
    """
    毛刺强度加权损失
    
    根据毛刺强度动态调整权重
    """
    
    def __init__(
        self,
        base_weight: float = 1.0,
        intensity_scale: float = 2.0,
        reduction: str = 'mean',
    ):
        super().__init__()
        self.base_weight = base_weight
        self.intensity_scale = intensity_scale
        self.reduction = reduction
        
    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        surge_labels: Optional[Dict[str, torch.Tensor]] = None,
        power_history: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            y_pred: 预测值
            y_true: 真实值
            surge_labels: 毛刺标签
            power_history: 历史功率（用于计算强度）
        """
        # 计算基础 MSE
        mse = F.mse_loss(y_pred, y_true, reduction='none')
        if mse.dim() > 1:
            mse = mse.mean(dim=-1)
        
        # 计算权重
        weights = torch.ones_like(mse) * self.base_weight
        
        if surge_labels is not None:
            # 使用预计算的毛刺强度
            if 'max_surge_intensity' in surge_labels and surge_labels['max_surge_intensity'] is not None:
                intensity = surge_labels['max_surge_intensity'].to(mse.device)
                
                # 强度归一化
                intensity_norm = intensity / (intensity.max() + 1e-8)
                
                # 强度加权
                weights = weights * (1.0 + self.intensity_scale * intensity_norm)
                
        elif power_history is not None:
            # 从历史功率计算毛刺强度
            diff = torch.abs(power_history[:, -1] - power_history[:, -2])
            intensity_norm = diff / (diff.max() + 1e-8)
            weights = weights * (1.0 + self.intensity_scale * intensity_norm)
        
        # 应用权重
        weighted_mse = mse * weights
        
        if self.reduction == 'mean':
            return weighted_mse.mean()
        elif self.reduction == 'sum':
            return weighted_mse.sum()
        else:
            return weighted_mse


# ============================================================================
# 第五部分：增强版综合损失函数
# ============================================================================

class EnhancedSurgeAwareCompoundLoss(nn.Module):
    """
    增强毛刺感知复合损失
    
    综合多种损失策略的最终方案，包含：
    1. 基础 MSE
    2. 毛刺感知 MSE
    3. 方向感知损失
    4. 强度加权损失
    5. 爬坡加权
    6. 波动加权
    7. 物理约束
    """
    
    def __init__(
        self,
        # 损失权重
        mse_weight: float = 1.0,
        surge_weight: float = 1.5,
        direction_weight: float = 0.5,
        intensity_weight: float = 0.5,
        
        # 爬坡加权
        use_ramp: bool = True,
        ramp_weight: float = 2.5,
        ramp_threshold: float = 0.1,
        
        # 波动加权
        use_volatility: bool = True,
        volatility_alpha: float = 2.0,
        
        # 物理约束
        use_physics: bool = True,
        lambda_smooth: float = 0.1,
        lambda_boundary: float = 0.2,
        
        # 归一化
        normalize_weights: bool = True,
    ):
        super().__init__()
        
        self.mse_weight = mse_weight
        self.surge_weight = surge_weight
        self.direction_weight = direction_weight
        self.intensity_weight = intensity_weight
        self.use_ramp = use_ramp
        self.use_volatility = use_volatility
        self.use_physics = use_physics
        
        # 损失函数
        self.surge_mse = SurgeAwareMSE(surge_weight=surge_weight, ramp_weight=ramp_weight)
        self.direction_loss_fn = DirectionAwareLoss(direction_weight=direction_weight)
        self.intensity_loss_fn = SurgeIntensityWeightedLoss(intensity_scale=2.0)
        
        # 爬坡损失
        if use_ramp:
            self.ramp_loss_fn = RampWeightedLoss(
                ramp_threshold=ramp_threshold,
                ramp_weight=ramp_weight,
            )
            
        # 波动损失
        if use_volatility:
            self.volatility_loss_fn = VolatilityWeightedLoss(alpha=volatility_alpha)
            
        # 物理损失
        if use_physics:
            self.physics_loss_fn = PhysicsGuidedCompoundLoss(
                lambda_smooth=lambda_smooth,
                lambda_boundary=lambda_boundary,
                lambda_ramp=ramp_weight,
            )
        
        # 权重配置
        self.loss_weights = {
            'mse': mse_weight,
            'surge': surge_weight,
            'direction': direction_weight,
            'intensity': intensity_weight,
            'ramp': ramp_weight if use_ramp else 0.0,
            'volatility': 0.3 if use_volatility else 0.0,
            'physics': 0.2 if use_physics else 0.0,
        }
        
    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        power_history: Optional[torch.Tensor] = None,
        surge_labels: Optional[Dict[str, torch.Tensor]] = None,
        volatility: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            y_pred: 预测值 (batch, horizon)
            y_true: 真实值 (batch, horizon)
            power_history: 历史功率 (batch, seq_len)
            surge_labels: 毛刺标签字典
            volatility: 波动率 (batch,)
            
        Returns:
            (总损失, 损失分解字典)
        """
        total_loss = torch.tensor(0.0, device=y_pred.device)
        losses = {}
        
        # 1. 基础 MSE
        mse = F.mse_loss(y_pred, y_true)
        total_loss = total_loss + self.mse_weight * mse
        losses['mse'] = mse.item()
        
        # 2. 毛刺感知 MSE
        surge_mse_loss = self.surge_mse(y_pred, y_true, surge_labels=surge_labels)
        total_loss = total_loss + self.surge_weight * surge_mse_loss
        losses['surge_mse'] = surge_mse_loss.item()
        
        # 3. 方向感知损失
        if self.direction_weight > 0:
            dir_loss = self.direction_loss_fn(
                y_pred, y_true,
                power_history=power_history,
                surge_labels=surge_labels,
            )
            total_loss = total_loss + self.direction_weight * dir_loss
            losses['direction'] = dir_loss.item()
        
        # 4. 强度加权损失
        if self.intensity_weight > 0:
            int_loss = self.intensity_loss_fn(
                y_pred, y_true,
                surge_labels=surge_labels,
                power_history=power_history,
            )
            total_loss = total_loss + self.intensity_weight * int_loss
            losses['intensity'] = int_loss.item()
        
        # 5. 爬坡加权损失
        if self.use_ramp and power_history is not None:
            ramp_loss = self.ramp_loss_fn(y_pred, y_true, power_history=power_history)
            total_loss = total_loss + self.loss_weights['ramp'] * ramp_loss
            losses['ramp'] = ramp_loss.item()
        
        # 6. 波动加权损失
        if self.use_volatility:
            vol_loss = self.volatility_loss_fn(y_pred, y_true, volatility=volatility)
            total_loss = total_loss + self.loss_weights['volatility'] * vol_loss
            losses['volatility'] = vol_loss.item()
        
        # 7. 物理引导损失
        if self.use_physics and power_history is not None:
            physics_loss, physics_parts = self.physics_loss_fn(
                y_pred, y_true, power_history=power_history
            )
            total_loss = total_loss + self.loss_weights['physics'] * physics_loss
            losses.update({f'physics_{k}': v for k, v in physics_parts.items()})
        
        return total_loss, losses


# ============================================================================
# 第六部分：训练辅助函数
# ============================================================================

def create_surge_labels_from_power(
    power_history: np.ndarray,
    power_future: np.ndarray,
    surge_threshold_ratio: float = 0.15,
    ramp_threshold_ratio: float = 0.1,
) -> Dict[str, np.ndarray]:
    """
    从功率数据创建毛刺标签
    
    Args:
        power_history: 历史功率 (N, seq_len)
        power_future: 目标功率 (N, horizon)
        surge_threshold_ratio: 毛刺阈值
        ramp_threshold_ratio: 爬坡阈值
        
    Returns:
        毛刺标签字典
    """
    # 检测毛刺
    if power_history.ndim == 1:
        power_history = power_history.reshape(1, -1)
    if power_future.ndim == 1:
        power_future = power_future.reshape(1, -1)
    
    n_samples = len(power_history)
    
    # 计算变化
    target_change = power_future[:, 0] - power_history[:, -1]
    power_max = power_history.max(axis=1)
    
    # 毛刺检测
    surge_threshold = surge_threshold_ratio * power_max
    ramp_threshold = ramp_threshold_ratio * power_max
    
    # 标签
    is_surge = np.abs(target_change) > surge_threshold * 2
    is_ramp = np.abs(target_change) > ramp_threshold
    
    # 主导方向
    dominant_direction = np.sign(target_change)
    
    # 最大强度
    max_intensity = np.abs(target_change)
    
    labels = {
        'is_surge_sample': is_surge.astype(float),
        'is_ramp': is_ramp.astype(float),
        'surge_direction': dominant_direction.astype(float),
        'max_surge_intensity': max_intensity,
    }
    
    return labels


def numpy_to_tensor_surge_labels(
    labels: Dict[str, np.ndarray],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    将 numpy 标签转换为 torch Tensor
    
    Args:
        labels: numpy 标签字典
        device: 目标设备
        
    Returns:
        torch Tensor 标签字典
    """
    tensor_labels = {}
    
    for key, value in labels.items():
        if value is not None:
            tensor_labels[key] = torch.FloatTensor(value).to(device)
    
    return tensor_labels


def create_weighted_sampler(
    power_history: np.ndarray,
    target_power: np.ndarray,
    volatility_calculator: Optional[VolatilityCalculator] = None,
    strategy: str = 'volatility',
    num_samples: int = None,
) -> WeightedRandomSampler:
    """
    创建加权采样器，用于DataLoader
    """
    if volatility_calculator is None:
        volatility_calculator = VolatilityCalculator()
    
    if strategy == 'volatility':
        weights = volatility_calculator.calculate_sample_weights(power_history, target_power)
    elif strategy == 'ramp':
        weights = volatility_calculator.calculate_ramp_weights(power_history)
    elif strategy == 'combined':
        vol_weights = volatility_calculator.calculate_sample_weights(power_history, target_power)
        ramp_weights = volatility_calculator.calculate_ramp_weights(power_history)
        weights = vol_weights * ramp_weights
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    # 归一化权重
    weights = weights / weights.sum() * len(weights)
    
    if num_samples is None:
        num_samples = len(weights)
    
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=num_samples,
        replacement=True,
    )
    
    return sampler


# ============================================================================
# 第七部分：训练工具函数
# ============================================================================

def train_with_surge_aware_loss(
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
    使用毛刺感知损失函数训练模型
    """
    if loss_config is None:
        loss_config = {
            'use_volatility': True,
            'use_ramp': True,
            'use_physics': True,
            'use_edge': True,
            'volatility_alpha': 2.0,
            'ramp_weight': 2.5,
        }
    
    if optimizer_config is None:
        optimizer_config = {
            'lr': 0.001,
            'weight_decay': 1e-5,
        }
    
    # 创建损失函数
    criterion = SurgeAwareCompoundLoss(**loss_config).to(device)
    
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
    
    # 训练循环
    history = {
        'train_loss': [],
        'val_loss': [],
        'loss_breakdown': [],
        'lr': [],
    }
    
    best_val_loss = float('inf')
    best_model_state = None
    no_improve = 0
    
    # 波动率计算器
    vol_calc = VolatilityCalculator()
    
    for epoch in range(max_epochs):
        # ----- 训练 -----
        model.train()
        train_losses = []
        train_loss_parts = []
        
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            # 获取历史功率用于加权
            if power_history_train is not None:
                start_idx = batch_idx * train_loader.batch_size
                end_idx = start_idx + len(X_batch)
                hist_idx = np.arange(start_idx, min(end_idx, len(power_history_train)))
                
                # 确保索引有效
                hist_idx = hist_idx[hist_idx < len(power_history_train)]
                if len(hist_idx) > 0:
                    power_hist = torch.FloatTensor(
                        power_history_train[hist_idx]
                    ).to(device)
                else:
                    power_hist = None
            else:
                power_hist = None
            
            optimizer.zero_grad()
            y_pred = model(X_batch)
            
            # 处理输出形状
            if y_pred.dim() == 3:
                y_pred = y_pred[:, -1, :]
            
            # 计算损失
            loss, loss_parts = criterion(y_pred, y_batch, power_history=power_hist)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
            if epoch == 0:
                train_loss_parts.append(loss_parts)
        
        # ----- 验证 -----
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch_idx, (X_batch, y_batch) in enumerate(val_loader):
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                
                if power_history_val is not None:
                    start_idx = batch_idx * val_loader.batch_size
                    end_idx = start_idx + len(X_batch)
                    hist_idx = np.arange(start_idx, min(end_idx, len(power_history_val)))
                    hist_idx = hist_idx[hist_idx < len(power_history_val)]
                    if len(hist_idx) > 0:
                        power_hist = torch.FloatTensor(
                            power_history_val[hist_idx]
                        ).to(device)
                    else:
                        power_hist = None
                else:
                    power_hist = None
                
                y_pred = model(X_batch)
                if y_pred.dim() == 3:
                    y_pred = y_pred[:, -1, :]
                
                loss, _ = criterion(y_pred, y_batch, power_history=power_hist)
                val_losses.append(loss.item())
        
        # 统计
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        if train_loss_parts:
            # 合并损失分解
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
        
        if (epoch + 1) % 5 == 0 or no_improve >= patience:
            print(f"Epoch {epoch+1:3d} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
                  f"Best: {best_val_loss:.4f}")
        
        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # 恢复最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, history


# ============================================================================
# 使用示例
# ============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("毛刺感知损失函数演示")
    print("=" * 60)
    
    # 模拟数据
    batch_size = 32
    seq_len = 12
    horizon = 4
    n_features = 8
    
    # 模拟数据
    X = torch.randn(batch_size, seq_len, n_features)
    y = torch.randn(batch_size, horizon)
    power_history = torch.randn(batch_size, seq_len) * 50 + 100  # 模拟发电功率
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X = X.to(device)
    y = y.to(device)
    power_history = power_history.to(device)
    
    print(f"\n设备: {device}")
    print(f"输入形状: {X.shape}")
    print(f"目标形状: {y.shape}")
    print(f"历史功率形状: {power_history.shape}")
    
    # 测试各种损失函数
    print("\n" + "-" * 40)
    print("1. 波动加权损失 (VolatilityWeightedLoss)")
    print("-" * 40)
    vol_loss_fn = VolatilityWeightedLoss(alpha=2.0)
    vol_loss = vol_loss_fn(y, y, volatility=torch.rand(batch_size).to(device))
    print(f"   损失值: {vol_loss.item():.4f}")
    
    print("\n" + "-" * 40)
    print("2. 爬坡加权损失 (RampWeightedLoss)")
    print("-" * 40)
    ramp_loss_fn = RampWeightedLoss(ramp_weight=3.0, ramp_threshold=0.1)
    ramp_loss = ramp_loss_fn(y, y, power_history=power_history)
    print(f"   损失值: {ramp_loss.item():.4f}")
    
    print("\n" + "-" * 40)
    print("3. 物理引导复合损失 (PhysicsGuidedCompoundLoss)")
    print("-" * 40)
    physics_loss_fn = PhysicsGuidedCompoundLoss(
        lambda_smooth=0.1,
        lambda_boundary=0.2,
        lambda_ramp=0.3,
    )
    physics_loss, physics_parts = physics_loss_fn(y, y, power_history=power_history)
    print(f"   总损失: {physics_loss.item():.4f}")
    print(f"   分解: {physics_parts}")
    
    print("\n" + "-" * 40)
    print("4. 基础综合损失 (SurgeAwareCompoundLoss)")
    print("-" * 40)
    surge_loss_fn = SurgeAwareCompoundLoss(
        use_volatility=True,
        use_ramp=True,
        use_physics=True,
        use_edge=True,
        volatility_alpha=2.0,
        ramp_weight=2.5,
        edge_weight=2.0,
    ).to(device)
    surge_loss, surge_parts = surge_loss_fn(y, y, power_history=power_history)
    print(f"   总损失: {surge_loss.item():.4f}")
    print(f"   分解: {surge_parts}")
    
    print("\n" + "-" * 40)
    print("5. 增强综合损失 (EnhancedSurgeAwareCompoundLoss)")
    print("-" * 40)
    enhanced_loss_fn = EnhancedSurgeAwareCompoundLoss(
        mse_weight=1.0,
        surge_weight=1.5,
        direction_weight=0.5,
        intensity_weight=0.5,
        use_ramp=True,
        use_volatility=True,
        use_physics=True,
    ).to(device)
    enhanced_loss, enhanced_parts = enhanced_loss_fn(y, y, power_history=power_history)
    print(f"   总损失: {enhanced_loss.item():.4f}")
    print(f"   分解: {enhanced_parts}")
    
    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)
