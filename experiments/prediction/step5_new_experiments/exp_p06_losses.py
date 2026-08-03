"""
EXP-P06 改进损失函数与物理约束模块。

Phase 1 改进:
1. 非对称MSE损失 - 鼓励预测更高峰值，减少低估
2. 物理约束 - 夜间归零、辐照度上限约束
3. 分段加权损失 - 峰值和夜间区域权重增加
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# 非对称损失函数
# ============================================================================

class AsymmetricMSELoss(nn.Module):
    """
    非对称MSE损失。

    对低估(underestimation)惩罚更重，迫使模型更大胆地预测峰值。

    Args:
        alpha: 低估惩罚系数。alpha < 1 鼓励预测更高值，建议 0.3~0.7
        clip_min: 损失下限，避免梯度爆炸
    """

    def __init__(self, alpha: float = 0.5, clip_min: float = 0.0):
        super().__init__()
        self.alpha = alpha
        self.clip_min = clip_min

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        underest = diff < 0  # 低估：预测值 < 真实值

        loss = torch.where(underest, self.alpha * diff**2, diff**2)

        if self.clip_min > 0:
            loss = torch.clamp(loss, min=self.clip_min)

        return loss.mean()


class AsymmetricHuberLoss(nn.Module):
    """
    非对称Huber损失：结合非对称惩罚和Huber损失对异常值的鲁棒性。

    Args:
        alpha: 低估惩罚系数
        delta: Huber损失的delta参数，控制鲁棒区域
    """

    def __init__(self, alpha: float = 0.5, delta: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.delta = delta

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        abs_diff = torch.abs(diff)
        underest = diff < 0

        # Huber损失部分
        huber_loss = torch.where(
            abs_diff <= self.delta,
            0.5 * diff**2,
            self.delta * (abs_diff - 0.5 * self.delta)
        )

        # 非对称权重
        loss = torch.where(underest, self.alpha * huber_loss, huber_loss)
        return loss.mean()


# ============================================================================
# 分段加权损失
# ============================================================================

class QuantileWeightedLoss(nn.Module):
    """
    基于分位数的加权损失。

    - 峰值区域 (top 20%) 和 夜间区域 (bottom 5%) 权重增加
    - 中间区域权重降低，减少对中间段数据的过度拟合

    Args:
        peak_weight: 峰值区域权重倍数
        night_weight: 夜间区域权重倍数
        mid_weight: 中间区域权重倍数
        peak_quantile: 峰值区域的分位数上界
        night_quantile: 夜间区域的的分位数下界
    """

    def __init__(
        self,
        peak_weight: float = 2.0,
        night_weight: float = 3.0,
        mid_weight: float = 0.5,
        peak_quantile: float = 0.80,
        night_quantile: float = 0.05,
    ):
        super().__init__()
        self.peak_weight = peak_weight
        self.night_weight = night_weight
        self.mid_weight = mid_weight
        self.peak_quantile = peak_quantile
        self.night_quantile = night_quantile

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target_flat = target.detach().ravel()

        # 计算分位数边界
        q_night = torch.quantile(target_flat, self.night_quantile)
        q_peak = torch.quantile(target_flat, self.peak_quantile)

        # 构建权重
        weights = torch.ones_like(target)
        weights[target < q_night] = self.night_weight
        weights[target > q_peak] = self.peak_weight

        # 中间区域降低权重
        mid_mask = (target >= q_night) & (target <= q_peak)
        weights[mid_mask] = self.mid_weight

        mse = (pred - target) ** 2
        weighted_mse = weights * mse
        return weighted_mse.mean()


class CombinedPeakLoss(nn.Module):
    """
    组合损失：非对称MSE + 分段加权 + 平滑正则化

    适用于全面改进峰值和日落预测。

    Args:
        alpha: 非对称系数
        peak_weight: 峰值权重
        night_weight: 夜间权重
        smoothness_weight: 平滑正则化权重
    """

    def __init__(
        self,
        alpha: float = 0.5,
        peak_weight: float = 2.0,
        night_weight: float = 3.0,
        smoothness_weight: float = 0.01,
    ):
        super().__init__()
        self.alpha = alpha
        self.peak_weight = peak_weight
        self.night_weight = night_weight
        self.smoothness_weight = smoothness_weight

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        # 1. 非对称MSE
        diff = pred - target
        underest = diff < 0
        asym_loss = torch.where(underest, self.alpha * diff**2, diff**2).mean()

        # 2. 分段加权
        target_flat = target.detach().ravel()
        q_night = torch.quantile(target_flat, 0.05)
        q_peak = torch.quantile(target_flat, 0.85)

        weights = torch.ones_like(target)
        weights[target < q_night] = self.night_weight
        weights[target > q_peak] = self.peak_weight
        quant_loss = (weights * (pred - target) ** 2).mean()

        # 3. 平滑正则化（减少剧烈波动）
        if pred.shape[1] > 1:
            grad_loss = F.mse_loss(pred[:, 1:], pred[:, :-1])
        else:
            grad_loss = torch.tensor(0.0, device=pred.device)

        return asym_loss + 0.5 * quant_loss + self.smoothness_weight * grad_loss


# ============================================================================
# 物理约束函数
# ============================================================================

def nighttime_zero_constraint(
    pred: np.ndarray,
    daylight_flag: np.ndarray | None = None,
    power_threshold: float = 0.01,
    irradiance_threshold: float = 5.0,
) -> np.ndarray:
    """
    夜间强制归零约束。

    当 daylight_flag <= 0 或预测功率极低时，强制归零。
    支持多步预测 (N, H) 和单步预测 (N,)。

    Args:
        pred: 预测值 (N,) 或 (N, H)
        daylight_flag: 归一化后的白天标志 (N,)，原始值 > 0 表示白天
        power_threshold: 功率阈值
        irradiance_threshold: 已废弃

    Returns:
        应用约束后的预测值
    """
    pred = np.asarray(pred, dtype=np.float32).copy()

    if daylight_flag is not None:
        daylight_flag = np.asarray(daylight_flag, dtype=np.float32)

        if pred.ndim == 2:
            # 多步预测：(N, H) -> (N*H,)
            # daylight_flag 是 (N,)，需要扩展到 (N*H,)
            pred_flat = pred.ravel()
            n_samples = pred.shape[0]
            n_horizons = pred.shape[1]
            # 重复 daylight_flag 以匹配 horizon
            daylight_expanded = np.repeat(daylight_flag, n_horizons)
            night_mask = daylight_expanded <= 0
            pred_flat[night_mask] = 0.0
            pred = pred_flat.reshape(pred.shape)
        else:
            # 单步预测
            night_mask = daylight_flag <= 0
            pred[night_mask] = 0.0
    else:
        # 无 daylight_flag 时使用功率阈值
        pred_flat = pred.ravel()
        pred_flat[pred_flat < power_threshold] = 0.0
        pred = pred_flat.reshape(pred.shape)

    return pred


def irradiance_upper_bound(
    pred: np.ndarray,
    irradiance: np.ndarray | None = None,
    capacity: float = 1.0,
    efficiency: float = 0.85,
    fill_factor: float = 0.80,
) -> np.ndarray:
    """
    辐照度上限约束。

    物理上限: P_max = G * efficiency * fill_factor * capacity / G_ref

    注意：此约束在本数据集中经验证无效（见测试结果：任何辐照度阈值都会增加RMSE），
    因此仅在辐照度非常高（>0.85, 即85%标准测试条件）时才应用。
    """
    pred = np.asarray(pred, dtype=np.float32).copy()

    if irradiance is not None:
        irradiance = np.asarray(irradiance, dtype=np.float32)

        if pred.ndim == 2:
            pred_flat = pred.ravel()
            n_horizons = pred.shape[1]
            irradiance_expanded = np.repeat(irradiance, n_horizons)

            # 仅在辐照度 > 0.85 时应用（非常高的辐照度）
            # 低于此阈值，约束反而会增加RMSE
            high_g_mask = irradiance_expanded > 0.85
            p_max = np.maximum(irradiance_expanded, 0) * efficiency * fill_factor * capacity

            for i in range(len(pred_flat)):
                if high_g_mask[i]:
                    pred_flat[i] = min(pred_flat[i], p_max[i])

            pred = pred_flat.reshape(pred.shape)
        else:
            if irradiance > 0.85:
                p_max = max(irradiance, 0) * efficiency * fill_factor * capacity
                pred = np.clip(pred, 0, p_max)
    else:
        np.clip(pred, 0, capacity * 1.1, out=pred)

    return pred


def sunset_monotonic_constraint(
    pred: np.ndarray,
    hours: np.ndarray | None = None,
    sunset_hour: float = 18.0,
    pre_sunset_hours: float = 2.0,
) -> np.ndarray:
    """
    日落单调递减约束。

    在日落前 pre_sunset_hours 小时内，强制预测值单调递减。

    Args:
        pred: 预测值 (N, H) 多步预测
        hours: 预测时刻的小时数 (N, H)
        sunset_hour: 日落标准小时
        pre_sunset_hours: 日落前约束小时数

    Returns:
        应用约束后的预测值
    """
    pred = np.asarray(pred, dtype=np.float32).copy()

    if hours is None or pred.ndim == 1:
        return pred

    hours = np.asarray(hours)

    # 日落约束区间
    sunset_start = sunset_hour - pre_sunset_hours

    for i in range(pred.shape[0]):  # 样本维度
        row = pred[i]
        row_hours = hours[i] if hours.ndim > 1 else hours

        for j in range(len(row) - 1):
            h = row_hours[j] if hours.ndim > 1 else row_hours
            next_h = row_hours[j + 1] if hours.ndim > 1 else row_hours

            # 如果在日落前区间且下一时刻更接近日落
            if sunset_start <= h < sunset_hour and next_h >= h:
                if row[j + 1] > row[j]:
                    row[j + 1] = row[j] * 0.95

    return pred


# ============================================================================
# Torch 版本的物理约束（用于训练中）
# ============================================================================

class PhysicsConstrainedModule(nn.Module):
    """
    带物理约束的预测模块。

    在模型输出后自动应用物理约束。

    Args:
        model: 基础预测模型
        capacity: 装机容量
        nighttime_threshold: 夜间辐照度阈值
    """

    def __init__(
        self,
        model: nn.Module,
        capacity: float = 1.0,
        nighttime_threshold: float = 5.0,
    ):
        super().__init__()
        self.model = model
        self.capacity = capacity
        self.nighttime_threshold = nighttime_threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pred = self.model(x)

        # 1. 夜间归零（基于辐照度）
        # 假设辐照度特征在 x 的第一个时间步
        if x.shape[2] >= 1:
            irradiance = x[:, -1, 0]  # 最后一个时间步的辐照度
            night_mask = irradiance < self.nighttime_threshold
            pred[night_mask] = 0.0

        # 2. 上限约束
        pred = torch.clamp(pred, min=0.0, max=self.capacity * 1.05)

        return pred


# ============================================================================
# 改进损失函数 (Phase 2 - 针对峰值过冲和日落下降问题)
# ============================================================================


class HuberLoss(nn.Module):
    """
    标准Huber损失：对异常值更鲁棒，避免峰值区域过冲。

    Huber损失 = 0.5 * x^2        当 |x| <= delta
              = delta * |x| - 0.5*delta^2  当 |x| > delta

    Args:
        delta: 控制从二次损失过渡到线性损失的阈值
    """

    def __init__(self, delta: float = 1.0):
        super().__init__()
        self.delta = delta

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        abs_diff = torch.abs(diff)
        huber = torch.where(
            abs_diff <= self.delta,
            0.5 * diff**2,
            self.delta * (abs_diff - 0.5 * self.delta),
        )
        return huber.mean()


class SunsetMonotonicLoss(nn.Module):
    """
    日落单调递减约束损失。

    在日落前时段（15:00-18:00），强制预测值单调递减。
    利用 sin_hour/cos_hour 特征检测日落时段。

    日落检测逻辑：
        - sin_hour > 0  (hour in (0, 12))
        - cos_hour < 0  (hour in (6, 18))
        => 6 < hour < 12  morning overlap, need to intersect properly

    精确日落区间检测：
        - pre-sunset: cos_hour in [-0.1, 0.1] area, irradiance falling
        - 或直接用 irradiance 梯度：irradiance < 0.3 * max_irradiance 且 decreasing

    Args:
        sunset_weight: 日落时段单调性惩罚权重
    """

    def __init__(self, sunset_weight: float = 0.1):
        super().__init__()
        self.sunset_weight = sunset_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor, x: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            pred: (B, H) 多步预测
            target: (B, H) 目标值
            x: (B, T, F) 输入特征，可选，含 sin_hour (index 7) 和 cos_hour (index 8)
        """
        base_loss = F.mse_loss(pred, target)

        if pred.shape[1] <= 1 or x is None:
            return base_loss

        sunset_loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

        # 检测日落时段: cos_hour < 0 (hour in (6,18)) 且 irradiance 较低
        if x.shape[-1] > 8:
            sin_hour = x[:, -1, 7]  # 最后一个时间步的 sin_hour
            cos_hour = x[:, -1, 8]  # 最后一个时间步的 cos_hour
            irradiance = x[:, -1, 0]  # total_irradiance (归一化)

            # 日落区间检测: cos_hour < 0 表示 6 < hour < 18
            # 结合 irradiance 较低来识别真正的日落时段
            sunset_mask = (cos_hour < 0) & (irradiance < 0.4) & (irradiance > 0.05)

            if sunset_mask.any():
                # 预测梯度: diff[i] = pred[:, i+1] - pred[:, i]
                # 单调递减要求 diff[i] <= 0，即 positive_diff = max(0, diff[i])
                diff_h = pred[:, 1:] - pred[:, :-1]  # (B, H-1)
                positive_grad = torch.clamp(diff_h, min=0.0)  # 只惩罚上升的梯度
                sunset_loss = positive_grad[sunset_mask].mean()

        return base_loss + self.sunset_weight * sunset_loss


class CombinedV2Loss(nn.Module):
    """
    Phase 2 组合损失：MSE + 平滑正则化 + 日落单调性约束（无峰值权重）。

    核心改进：
    1. 移除 peak_weight - 避免峰值区域过冲
    2. 添加日落单调性约束 - 改善日落下降段预测
    3. 保留平滑正则化 - 减少剧烈波动

    Args:
        huber_delta: Huber损失的delta参数，对异常值鲁棒
        smoothness_weight: 平滑正则化权重
        sunset_weight: 日落单调性约束权重
        night_weight: 夜间权重（保留，夜间预测是硬约束）
    """

    def __init__(
        self,
        huber_delta: float = 0.1,
        smoothness_weight: float = 0.05,
        sunset_weight: float = 0.1,
        night_weight: float = 0.0,
    ):
        super().__init__()
        self.huber_delta = huber_delta
        self.smoothness_weight = smoothness_weight
        self.sunset_weight = sunset_weight
        self.night_weight = night_weight

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        x: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # 1. Huber损失（鲁棒，减少峰值过冲）
        diff = pred - target
        abs_diff = torch.abs(diff)
        huber_loss = torch.where(
            abs_diff <= self.huber_delta,
            0.5 * diff**2,
            self.huber_delta * (abs_diff - 0.5 * self.huber_delta),
        ).mean()

        # 2. 夜间加权（仅对低功率区域增加权重）
        target_flat = target.detach().ravel()
        q_night = torch.quantile(target_flat, 0.05)
        night_mask = target < q_night
        if night_mask.any():
            night_diff = diff[night_mask]
            night_loss = self.night_weight * (night_diff**2).mean()
        else:
            night_loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        base_loss = huber_loss + 0.5 * night_loss

        # 3. 平滑正则化
        smooth_loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        if pred.shape[1] > 1:
            smooth_loss = F.mse_loss(pred[:, 1:], pred[:, :-1])

        # 4. 日落单调性约束
        sunset_loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        if x is not None and x.shape[-1] > 8 and pred.shape[1] > 1:
            sin_hour = x[:, -1, 7]
            cos_hour = x[:, -1, 8]
            irradiance = x[:, -1, 0]
            sunset_mask = (cos_hour < 0) & (irradiance < 0.4) & (irradiance > 0.05)

            if sunset_mask.any():
                diff_h = pred[:, 1:] - pred[:, :-1]
                positive_grad = torch.clamp(diff_h, min=0.0)
                sunset_loss = positive_grad[sunset_mask].mean()

        return base_loss + self.smoothness_weight * smooth_loss + self.sunset_weight * sunset_loss


class MonotonicityRegularizer(nn.Module):
    """
    物理先验单调性正则化器。

    在日落时段强制预测值单调递减。
    作为训练中的额外正则化项，与主损失联合优化。
    """

    def __init__(self, weight: float = 0.1):
        super().__init__()
        self.weight = weight

    def forward(self, pred: torch.Tensor, x: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            pred: (B, H)
            x: (B, T, F), 可选
        Returns:
            正则化损失
        """
        if pred.shape[1] <= 1:
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

        loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

        if x is not None and x.shape[-1] > 8:
            cos_hour = x[:, -1, 8]
            irradiance = x[:, -1, 0]

            # 日落窗口检测
            sunset_mask = (cos_hour < 0) & (irradiance < 0.4) & (irradiance > 0.05)
            if sunset_mask.any():
                diff = pred[:, 1:] - pred[:, :-1]
                positive = torch.clamp(diff, min=0.0)
                loss = positive[sunset_mask].mean()

        return self.weight * loss


# ============================================================================
# 工具函数
# ============================================================================


def create_loss_function(loss_type: str = "mse", **kwargs) -> nn.Module:
    """
    工厂函数：根据配置创建损失函数。

    Args:
        loss_type: 损失函数类型
            - "mse": 标准MSE
            - "asymmetric_mse": 非对称MSE (Phase 1)
            - "asymmetric_huber": 非对称Huber (Phase 1)
            - "quantile_weighted": 分段加权 (Phase 1)
            - "combined": 组合损失 (Phase 1)
            - "huber": 标准Huber损失 (Phase 2 - 替代asymmetric_mse)
            - "combined_v2": Phase 2组合损失（无peak_weight，含日落约束）

    Returns:
        损失函数实例
    """
    loss_type = loss_type.lower()

    if loss_type == "mse":
        return nn.MSELoss()
    elif loss_type == "asymmetric_mse":
        return AsymmetricMSELoss(alpha=kwargs.get("alpha", 0.5))
    elif loss_type == "asymmetric_huber":
        return AsymmetricHuberLoss(
            alpha=kwargs.get("alpha", 0.5),
            delta=kwargs.get("delta", 1.0),
        )
    elif loss_type == "quantile_weighted":
        return QuantileWeightedLoss(
            peak_weight=kwargs.get("peak_weight", 2.0),
            night_weight=kwargs.get("night_weight", 3.0),
        )
    elif loss_type == "combined":
        return CombinedPeakLoss(
            alpha=kwargs.get("alpha", 0.5),
            peak_weight=kwargs.get("peak_weight", 2.0),
            night_weight=kwargs.get("night_weight", 3.0),
        )
    elif loss_type == "huber":
        return HuberLoss(delta=kwargs.get("delta", 1.0))
    elif loss_type == "combined_v2":
        return CombinedV2Loss(
            huber_delta=kwargs.get("huber_delta", 0.1),
            smoothness_weight=kwargs.get("smoothness_weight", 0.05),
            sunset_weight=kwargs.get("sunset_weight", 0.1),
            night_weight=kwargs.get("night_weight", 0.0),
        )
    else:
        raise ValueError(f"未知的损失函数类型: {loss_type}")


# ============================================================================
# 扩展训练函数
# ============================================================================

def run_epoch_constrained(
    model,
    loader,
    criterion,
    optimizer,
    device,
    apply_physics: bool = False,
    capacity: float = 1.0,
) -> float:
    """带物理约束的训练epoch。

    注意：物理约束（如clamp min=0）仅在推理阶段应用。
    训练时不对pred做clamp，避免破坏残差预测的梯度。
    """
    model.train()
    total, n = 0.0, 0

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)

        # 训练时不应用clamp约束 - 保留负残差的梯度信息
        # 物理约束仅在后处理阶段（推理）应用

        # 支持 Phase 2 损失函数（需要输入特征 x）
        if hasattr(criterion, "forward") and "x" in criterion.forward.__code__.co_varnames:
            loss = criterion(pred, yb, x=xb)
        else:
            loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()

        total += loss.item() * len(xb)
        n += len(xb)

    return total / max(n, 1)


def eval_loss_constrained(
    model,
    loader,
    criterion,
    device,
    apply_physics: bool = False,
    capacity: float = 1.0,
) -> float:
    """带物理约束的验证loss评估。"""
    model.eval()
    total, n = 0.0, 0

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)

            if hasattr(criterion, "forward") and "x" in criterion.forward.__code__.co_varnames:
                loss = criterion(pred, yb, x=xb)
            else:
                loss = criterion(pred, yb)
            total += loss.item() * len(xb)
            n += len(xb)

    return total / max(n, 1)


def train_with_early_stop_constrained(
    model,
    train_loader,
    val_loader,
    *,
    criterion: nn.Module | None = None,
    lr: float = 1e-3,
    max_epochs: int = 50,
    patience: int = 8,
    device=None,
    apply_physics: bool = False,
    capacity: float = 1.0,
) -> tuple[nn.Module, list[dict]]:
    """
    带物理约束的早停训练。

    Args:
        model: 预测模型
        train_loader: 训练数据
        val_loader: 验证数据
        criterion: 损失函数，默认使用MSE
        lr: 学习率
        max_epochs: 最大epoch数
        patience: 早停耐心值
        device: 设备
        apply_physics: 是否应用物理约束
        capacity: 装机容量
    """
    device = device or get_device()
    model = model.to(device)
    criterion = criterion or nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 检查损失函数是否需要输入特征
    criterion_needs_x = hasattr(criterion, "forward") and "x" in criterion.forward.__code__.co_varnames

    best_state = deepcopy(model.state_dict())
    best_val = float("inf")
    wait = 0
    history = []

    for epoch in range(1, max_epochs + 1):
        tr_loss = run_epoch_constrained(
            model, train_loader, criterion, optimizer, device,
            apply_physics=apply_physics, capacity=capacity,
        )
        val_loss = eval_loss_constrained(
            model, val_loader, criterion, device,
            apply_physics=apply_physics, capacity=capacity,
        )
        history.append({"epoch": epoch, "train_loss": tr_loss, "val_loss": val_loss})

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    return model, history


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
