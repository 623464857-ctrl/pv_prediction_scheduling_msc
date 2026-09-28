"""
毛刺特征工程模块 (Surge Feature Engineering)

提供毛刺检测和特征提取功能，用于：
1. 从原始功率序列中检测毛刺时刻
2. 提取毛刺相关特征
3. 生成毛刺掩码和标签

Author: AI Assistant
Date: 2026-09-27
"""

import numpy as np
from typing import Tuple, Optional, Dict
from scipy import signal
from scipy.ndimage import uniform_filter1d


class SurgeFeatureExtractor:
    """
    毛刺特征提取器
    
    从时间序列中检测和提取毛刺特征：
    - 毛刺检测（基于梯度阈值）
    - 毛刺类型分类（骤升/骤降）
    - 毛刺强度计算
    - 毛刺掩码生成
    """
    
    def __init__(
        self,
        surge_threshold_ratio: float = 0.15,
        min_surge_width: int = 1,
        max_surge_width: int = 10,
        smooth_window: int = 3,
        use_adaptive_threshold: bool = True,
    ):
        """
        Args:
            surge_threshold_ratio: 毛刺阈值（相对于最大功率的比例）
            min_surge_width: 最小毛刺宽度（样本数）
            max_surge_width: 最大毛刺宽度（样本数）
            smooth_window: 平滑窗口大小
            use_adaptive_threshold: 是否使用自适应阈值
        """
        self.surge_threshold_ratio = surge_threshold_ratio
        self.min_surge_width = min_surge_width
        self.max_surge_width = max_surge_width
        self.smooth_window = smooth_window
        self.use_adaptive_threshold = use_adaptive_threshold
        
    def detect_surges(
        self,
        power: np.ndarray,
        threshold: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        检测功率序列中的毛刺
        
        Args:
            power: 功率序列 (N,) 或 (N, seq_len)
            threshold: 固定阈值，如果为None则使用自适应阈值
            
        Returns:
            (surge_mask, surge_direction, surge_magnitude)
            - surge_mask: 毛刺点掩码 (N,) 或 (N, seq_len)
            - surge_direction: 毛刺方向 (+1=骤升, -1=骤降, 0=无)
            - surge_magnitude: 毛刺强度 (N,) 或 (N, seq_len)
        """
        if power.ndim == 1:
            power = power.reshape(-1, 1)
            squeeze_output = True
        else:
            squeeze_output = False
            
        n_samples, seq_len = power.shape
        
        # 计算一阶差分
        diff = np.diff(power, axis=1, prepend=power[:, :1])
        
        # 计算阈值
        if threshold is None:
            if self.use_adaptive_threshold:
                # 自适应阈值：基于局部统计
                power_std = np.std(power, axis=1, keepdims=True)
                threshold = self.surge_threshold_ratio * power.max(axis=1, keepdims=True)
                threshold = np.maximum(threshold, 2 * power_std)
            else:
                threshold = self.surge_threshold_ratio * power.max()
        
        # 检测毛刺
        surge_mask = np.abs(diff) > threshold
        surge_direction = np.sign(diff)
        surge_magnitude = np.abs(diff)
        
        # 过滤太窄的毛刺（去除噪声）
        surge_mask = self._filter_short_surges(surge_mask)
        
        # 平滑处理
        if self.smooth_window > 1:
            surge_mask = self._smooth_surge_mask(surge_mask)
        
        if squeeze_output:
            surge_mask = surge_mask.squeeze()
            surge_direction = surge_direction.squeeze()
            surge_magnitude = surge_magnitude.squeeze()
            
        return surge_mask, surge_direction, surge_magnitude
    
    def _filter_short_surges(self, surge_mask: np.ndarray) -> np.ndarray:
        """过滤持续时间过短的毛刺"""
        n_samples, seq_len = surge_mask.shape
        filtered = surge_mask.copy()
        
        for i in range(n_samples):
            # 找到毛刺段的起止位置
            mask = surge_mask[i]
            diff_mask = np.diff(np.concatenate([[0], mask, [0]]).astype(int))
            starts = np.where(diff_mask == 1)[0]
            ends = np.where(diff_mask == -1)[0]
            
            for start, end in zip(starts, ends):
                width = end - start
                if width < self.min_surge_width:
                    filtered[i, start:end] = 0
                    
        return filtered
    
    def _smooth_surge_mask(self, surge_mask: np.ndarray) -> np.ndarray:
        """平滑毛刺掩码"""
        smoothed = np.zeros_like(surge_mask)
        for i in range(surge_mask.shape[0]):
            smoothed[i] = uniform_filter1d(
                surge_mask[i].astype(float),
                size=self.smooth_window,
                mode='constant',
            ) > 0.3
        return smoothed
    
    def extract_surge_features(
        self,
        power_history: np.ndarray,
        power_future: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """
        提取毛刺相关特征
        
        Args:
            power_history: 历史功率 (N, seq_len) 或 (N,)
            power_future: 未来功率 (N, horizon) 或 (N,)，用于标签
            
        Returns:
            特征字典:
            - surge_mask: 毛刺掩码
            - surge_direction: 毛刺方向
            - surge_magnitude: 毛刺强度
            - surge_position: 毛刺位置（相对索引）
            - is_surge_sample: 样本级别毛刺标签
            - surge_count: 毛刺点数量
            - max_surge_intensity: 最大毛刺强度
        """
        if power_history.ndim == 1:
            power_history = power_history.reshape(1, -1)
            
        n_samples, seq_len = power_history.shape
        
        # 检测毛刺
        surge_mask, surge_direction, surge_magnitude = self.detect_surges(power_history)
        
        # 样本级别毛刺标签
        is_surge_sample = surge_mask.any(axis=1).astype(float)
        
        # 毛刺计数
        surge_count = surge_mask.sum(axis=1)
        
        # 最大毛刺强度
        max_surge_intensity = np.zeros(n_samples)
        for i in range(n_samples):
            if surge_mask[i].any():
                max_surge_intensity[i] = surge_magnitude[i][surge_mask[i]].max()
            else:
                max_surge_intensity[i] = 0.0
        
        # 毛刺位置（最后一个毛刺的相对位置）
        surge_position = np.zeros(n_samples)
        for i in range(n_samples):
            surge_indices = np.where(surge_mask[i])[0]
            if len(surge_indices) > 0:
                surge_position[i] = surge_indices[-1] / seq_len  # 归一化到 [0, 1]
            else:
                surge_position[i] = -1  # 无毛刺
                
        # 主导毛刺方向
        dominant_direction = np.zeros(n_samples)
        for i in range(n_samples):
            surge_diffs = surge_direction[i][surge_mask[i]]
            if len(surge_diffs) > 0:
                dominant_direction[i] = 1 if np.mean(surge_diffs) > 0 else -1
            else:
                dominant_direction[i] = 0
                
        features = {
            'surge_mask': surge_mask,
            'surge_direction': surge_direction,
            'surge_magnitude': surge_magnitude,
            'surge_position': surge_position,
            'is_surge_sample': is_surge_sample,
            'surge_count': surge_count,
            'max_surge_intensity': max_surge_intensity,
            'dominant_direction': dominant_direction,
        }
        
        # 如果有未来功率，计算预测目标的爬坡标签
        if power_future is not None:
            if power_future.ndim == 1:
                power_future = power_future.reshape(-1, 1)
                
            # 计算目标变化
            target_change = power_future[:, 0] - power_history[:, -1]
            target_ramp_up = (target_change > 0).astype(float)
            target_ramp_down = (target_change < 0).astype(float)
            
            features['target_ramp_up'] = target_ramp_up
            features['target_ramp_down'] = target_ramp_down
            features['target_ramp_magnitude'] = np.abs(target_change)
            
        return features
    
    def create_surge_labels(
        self,
        power_history: np.ndarray,
        power_future: np.ndarray,
        ramp_threshold_ratio: float = 0.1,
    ) -> Dict[str, np.ndarray]:
        """
        创建用于训练的毛刺标签
        
        Args:
            power_history: 历史功率 (N, seq_len)
            power_future: 目标功率 (N, horizon)
            ramp_threshold_ratio: 爬坡阈值
            
        Returns:
            标签字典:
            - is_ramp: 是否为爬坡样本
            - ramp_direction: 爬坡方向 (+1, -1, 0)
            - ramp_magnitude: 爬坡幅度
            - is_surge: 是否为毛刺样本（大幅爬坡）
        """
        if power_history.ndim == 1:
            power_history = power_history.reshape(1, -1)
        if power_future.ndim == 1:
            power_future = power_future.reshape(1, 1)
            
        n_samples = len(power_history)
        horizon = power_future.shape[1]
        
        # 计算目标变化
        target_change = power_future[:, 0] - power_history[:, -1]  # 第一个预测步的变化
        
        # 计算阈值
        power_max = power_history.max(axis=1)
        ramp_threshold = ramp_threshold_ratio * power_max
        
        # 爬坡标签
        is_ramp = np.abs(target_change) > ramp_threshold
        ramp_direction = np.sign(target_change)
        ramp_magnitude = np.abs(target_change)
        
        # 毛刺标签（大幅爬坡）
        surge_threshold = 2 * ramp_threshold
        is_surge = np.abs(target_change) > surge_threshold
        
        return {
            'is_ramp': is_ramp.astype(float),
            'ramp_direction': ramp_direction.astype(float),
            'ramp_magnitude': ramp_magnitude,
            'is_surge': is_surge.astype(float),
            'surge_direction': ramp_direction * is_surge.astype(float),
        }
    
    def get_surge_weights(
        self,
        power_history: np.ndarray,
        power_future: np.ndarray,
        base_weight: float = 1.0,
        surge_weight: float = 3.0,
        ramp_weight: float = 2.0,
    ) -> np.ndarray:
        """
        计算样本训练权重
        
        Args:
            power_history: 历史功率
            power_future: 目标功率
            base_weight: 基础权重
            surge_weight: 毛刺样本权重倍数
            ramp_weight: 爬坡样本权重倍数
            
        Returns:
            样本权重 (N,)
        """
        labels = self.create_surge_labels(power_history, power_future)
        
        weights = np.ones(len(power_history)) * base_weight
        weights[labels['is_surge'].astype(bool)] *= surge_weight
        weights[labels['is_ramp'].astype(bool)] *= ramp_weight
        
        # 额外增强：基于爬坡幅度
        ramp_magnitude_normalized = labels['ramp_magnitude'] / (labels['ramp_magnitude'].max() + 1e-8)
        weights = weights * (1.0 + 0.5 * ramp_magnitude_normalized)
        
        return weights


class SurgeFeatureInjector:
    """
    毛刺特征注入器
    
    将毛刺特征注入到模型输入或中间层
    """
    
    def __init__(
        self,
        surge_extractor: Optional[SurgeFeatureExtractor] = None,
        inject_mode: str = 'concat',  # 'concat', 'add', 'gate'
        surge_embed_dim: int = 8,
    ):
        """
        Args:
            surge_extractor: 毛刺特征提取器
            inject_mode: 注入模式 ('concat'=拼接, 'add'=相加, 'gate'=门控)
            surge_embed_dim: 毛刺嵌入维度
        """
        self.surge_extractor = surge_extractor or SurgeFeatureExtractor()
        self.inject_mode = inject_mode
        self.surge_embed_dim = surge_embed_dim
        
    def inject_to_features(
        self,
        X: np.ndarray,
        power_history: np.ndarray,
        mode: str = 'sample',  # 'sample'=样本级, 'seq'=序列级
    ) -> np.ndarray:
        """
        将毛刺特征注入到特征矩阵
        
        Args:
            X: 原始特征 (N, seq_len, n_features)
            power_history: 历史功率 (N, seq_len)
            mode: 注入模式 ('sample'=样本级广播, 'seq'=序列级)
            
        Returns:
            增强后的特征 (N, seq_len, n_features + surge_features)
        """
        # 提取毛刺特征
        features = self.surge_extractor.extract_surge_features(power_history)
        
        # 选择要注入的特征
        # 确保所有特征都是 (N, seq_len) 形状
        surge_dir = features['surge_direction']
        surge_mag = features['surge_magnitude']
        is_surge = features['is_surge_sample']
        
        # 广播 is_surge_sample 到 (N, seq_len)
        if is_surge.ndim == 1:
            is_surge = np.broadcast_to(
                is_surge[:, np.newaxis],
                (is_surge.shape[0], surge_dir.shape[1] if surge_dir.ndim > 1 else surge_dir.shape[0])
            )
        
        surge_features = np.stack([
            surge_dir,      # 方向
            surge_mag,      # 强度
            is_surge,       # 毛刺标签
        ], axis=-1)  # (N, seq_len, 3)
        
        if mode == 'sample':
            # 广播到所有时间步（已经是 seq 级别）
            pass  # surge_features 已经是 (N, seq_len, 3)
                
        # 拼接
        X_enhanced = np.concatenate([X, surge_features], axis=-1)
        
        return X_enhanced
    
    def create_surge_channel(
        self,
        power_history: np.ndarray,
        seq_len: int,
    ) -> np.ndarray:
        """
        创建毛刺通道（用于与输入拼接）
        
        Args:
            power_history: 历史功率 (N, seq_len)
            seq_len: 目标序列长度
            
        Returns:
            毛刺通道 (N, seq_len, 1)
        """
        features = self.surge_extractor.extract_surge_features(power_history)
        
        # 创建归一化的毛刺强度通道
        surge_channel = np.abs(features['surge_direction']) * features['surge_magnitude']
        
        # 归一化
        surge_channel = surge_channel / (surge_channel.max() + 1e-8)
        
        return surge_channel.unsqueeze(-1) if isinstance(surge_channel, np.ndarray) and surge_channel.ndim == 2 else surge_channel


def create_surge_aware_dataset(
    X: np.ndarray,
    y: np.ndarray,
    power_history: np.ndarray,
    surge_threshold_ratio: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    创建支持毛刺感知学习的增强数据集
    
    Args:
        X: 原始特征 (N, seq_len, n_features)
        y: 目标值 (N, horizon) 或 (N,)
        power_history: 历史功率 (N, seq_len)
        surge_threshold_ratio: 毛刺阈值
        
    Returns:
        (X_enhanced, y, sample_weights, surge_info)
    """
    extractor = SurgeFeatureExtractor(surge_threshold_ratio=surge_threshold_ratio)
    
    # 提取毛刺特征
    surge_features = extractor.extract_surge_features(power_history, power_future=y)
    
    # 计算样本权重
    weights = extractor.get_surge_weights(power_history, y)
    
    # 增强特征
    injector = SurgeFeatureInjector(extractor)
    X_enhanced = injector.inject_to_features(X, power_history, mode='sample')
    
    # 毛刺信息
    surge_info = {
        'is_surge_sample': surge_features['is_surge_sample'],
        'surge_count': surge_features['surge_count'],
        'max_surge_intensity': surge_features['max_surge_intensity'],
        'dominant_direction': surge_features['dominant_direction'],
        'target_ramp_up': surge_features.get('target_ramp_up', None),
        'target_ramp_down': surge_features.get('target_ramp_down', None),
    }
    
    return X_enhanced, y, weights, surge_info


# ============================================================================
# 辅助函数
# ============================================================================

def analyze_surge_patterns(
    power: np.ndarray,
    timestamps: Optional[np.ndarray] = None,
    surge_extractor: Optional[SurgeFeatureExtractor] = None,
) -> Dict:
    """
    分析功率序列中的毛刺模式
    
    Args:
        power: 功率序列 (N,) 或 (N, seq_len)
        timestamps: 时间戳（可选）
        surge_extractor: 毛刺提取器
        
    Returns:
        分析结果字典
    """
    if surge_extractor is None:
        surge_extractor = SurgeFeatureExtractor()
        
    features = surge_extractor.extract_surge_features(power)
    
    results = {
        'n_surge_samples': int(features['is_surge_sample'].sum()),
        'surge_ratio': float(features['is_surge_sample'].mean()),
        'avg_surge_count': float(features['surge_count'].mean()),
        'max_surge_intensity': float(features['max_surge_intensity'].max()),
        'surge_direction_dist': {
            'up': int((features['dominant_direction'] > 0).sum()),
            'down': int((features['dominant_direction'] < 0).sum()),
            'none': int((features['dominant_direction'] == 0).sum()),
        },
    }
    
    return results


# ============================================================================
# 使用示例
# ============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("毛刺特征工程模块演示")
    print("=" * 60)
    
    # 模拟发电功率数据
    np.random.seed(42)
    n_samples = 100
    seq_len = 16
    
    # 模拟正常功率 + 毛刺
    base_power = np.random.uniform(50, 100, (n_samples, seq_len))
    
    # 添加毛刺
    surge_indices = np.random.choice(n_samples, 20, replace=False)
    for idx in surge_indices:
        # 骤升
        if np.random.random() > 0.5:
            base_power[idx, -3:] = base_power[idx, -3:] + np.random.uniform(30, 50)
        else:
            # 骤降
            base_power[idx, -3:] = base_power[idx, -3:] - np.random.uniform(30, 50)
    
    # 模拟目标功率
    target_power = base_power[:, -1:] + np.random.uniform(-10, 10, (n_samples, 1))
    
    print(f"\n数据形状: power={base_power.shape}, target={target_power.shape}")
    
    # 提取毛刺特征
    extractor = SurgeFeatureExtractor(surge_threshold_ratio=0.15)
    features = extractor.extract_surge_features(base_power, power_future=target_power)
    
    print(f"\n毛刺检测结果:")
    print(f"  毛刺样本数: {int(features['is_surge_sample'].sum())}/{n_samples}")
    print(f"  平均毛刺数: {features['surge_count'].mean():.2f}")
    print(f"  最大毛刺强度: {features['max_surge_intensity'].max():.2f}")
    
    # 分析毛刺模式
    patterns = analyze_surge_patterns(base_power, surge_extractor=extractor)
    print(f"\n毛刺模式分析:")
    print(f"  毛刺比例: {patterns['surge_ratio']*100:.1f}%")
    print(f"  方向分布: {patterns['surge_direction_dist']}")
    
    # 计算权重
    weights = extractor.get_surge_weights(base_power, target_power)
    print(f"\n样本权重统计:")
    print(f"  权重范围: [{weights.min():.2f}, {weights.max():.2f}]")
    print(f"  平均权重: {weights.mean():.2f}")
    
    # 创建增强数据集
    X_dummy = np.random.randn(n_samples, seq_len, 8)
    X_enhanced, y, sample_weights, surge_info = create_surge_aware_dataset(
        X_dummy, target_power, base_power
    )
    
    print(f"\n增强数据集:")
    print(f"  原始特征维度: {X_dummy.shape[-1]}")
    print(f"  增强后特征维度: {X_enhanced.shape[-1]}")
    print(f"  样本权重范围: [{sample_weights.min():.2f}, {sample_weights.max():.2f}]")
    
    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)
