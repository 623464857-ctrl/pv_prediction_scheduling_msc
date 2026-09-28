"""
毛刺感知样本准备脚本 (Surge-Aware Sample Preparation)

准备支持毛刺特征的数据集：
1. 从原始数据中提取毛刺特征
2. 将毛刺特征与主特征合并
3. 生成毛刺标签用于训练
4. 保存增强后的样本数据

Author: AI Assistant
Date: 2026-09-27
"""

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.prediction.step3_deep_learning.exp_03_surge_features import (
    SurgeFeatureExtractor,
    create_surge_aware_dataset,
    analyze_surge_patterns,
)


# ============================================================================
# 配置
# ============================================================================

DEFAULT_CONFIG = {
    # 毛刺检测参数
    'surge_threshold_ratio': 0.15,
    'min_surge_width': 1,
    'max_surge_width': 10,
    'smooth_window': 3,
    'use_adaptive_threshold': True,
    
    # 样本准备参数
    'ramp_threshold_ratio': 0.1,
    'surge_weight': 3.0,
    'ramp_weight': 2.0,
    'base_weight': 1.0,
}


# ============================================================================
# 核心函数
# ============================================================================

def prepare_surge_features(
    df: pd.DataFrame,
    power_col: str = 'power',
    time_col: str = 'timestamp',
    feature_cols: Optional[list] = None,
    lookback: int = 16,
    horizon: int = 1,
    surge_threshold_ratio: float = 0.15,
) -> Tuple[Dict, Dict]:
    """
    准备毛刺特征数据集
    
    Args:
        df: 原始数据 DataFrame
        power_col: 功率列名
        time_col: 时间列名
        feature_cols: 特征列名列表（不包括功率）
        lookback: 回看步长
        horizon: 预测步长
        surge_threshold_ratio: 毛刺检测阈值
        
    Returns:
        (特征数据字典, 毛刺分析字典)
    """
    # 排序
    df = df.sort_values(time_col).reset_index(drop=True)
    
    # 提取功率序列
    power = df[power_col].values
    
    # 确定特征列
    if feature_cols is None:
        feature_cols = [col for col in df.columns 
                       if col not in [power_col, time_col] and df[col].dtype in [np.float64, np.int64]]
    
    # 创建特征提取器
    extractor = SurgeFeatureExtractor(
        surge_threshold_ratio=surge_threshold_ratio,
        min_surge_width=DEFAULT_CONFIG['min_surge_width'],
        smooth_window=DEFAULT_CONFIG['smooth_window'],
        use_adaptive_threshold=DEFAULT_CONFIG['use_adaptive_threshold'],
    )
    
    # 构建序列数据
    n_total = len(df)
    n_samples = n_total - lookback - horizon + 1
    
    if n_samples <= 0:
        raise ValueError(f"数据量不足: 需要至少 {lookback + horizon} 个样本")
    
    # 初始化
    X_list = []
    y_list = []
    power_history_list = []
    surge_features_list = []
    surge_labels_list = []
    
    for i in range(n_samples):
        # 历史窗口
        start_idx = i
        end_idx = i + lookback
        
        # 预测窗口
        pred_start = end_idx
        pred_end = end_idx + horizon
        
        # 提取特征
        X_seq = df[feature_cols].iloc[start_idx:end_idx].values
        y_target = df[power_col].iloc[pred_start:pred_end].values
        power_hist = df[power_col].iloc[start_idx:end_idx].values
        
        X_list.append(X_seq)
        y_list.append(y_target)
        power_history_list.append(power_hist)
        
        # 提取毛刺特征
        surge_features = extractor.extract_surge_features(
            power_hist.reshape(1, -1),
            power_future=y_target.reshape(1, -1)
        )
        surge_features_list.append({
            'surge_direction': surge_features['surge_direction'].squeeze(),
            'surge_magnitude': surge_features['surge_magnitude'].squeeze(),
            'surge_mask': surge_features['surge_mask'].squeeze(),
        })
        
        # 创建毛刺标签
        surge_labels = extractor.create_surge_labels(
            power_hist.reshape(1, -1),
            y_target.reshape(1, -1),
            ramp_threshold_ratio=DEFAULT_CONFIG['ramp_threshold_ratio'],
        )
        surge_labels_list.append(surge_labels)
    
    # 转换为数组
    X = np.stack(X_list, axis=0)  # (N, lookback, n_features)
    y = np.stack(y_list, axis=0)  # (N, horizon)
    power_history = np.stack(power_history_list, axis=0)  # (N, lookback)
    
    # 构建毛刺特征数组
    surge_direction = np.stack([s['surge_direction'] for s in surge_features_list], axis=0)
    surge_magnitude = np.stack([s['surge_magnitude'] for s in surge_features_list], axis=0)
    surge_mask = np.stack([s['surge_mask'] for s in surge_features_list], axis=0)
    
    # 合并毛刺特征
    surge_features = np.stack([
        surge_direction,
        surge_magnitude / (surge_magnitude.max() + 1e-8),  # 归一化
        surge_mask.astype(float),
    ], axis=-1)  # (N, lookback, 3)
    
    # 构建样本权重
    sample_weights = np.zeros(len(X))
    for i, labels in enumerate(surge_labels_list):
        weight = DEFAULT_CONFIG['base_weight']
        if labels['is_surge']:
            weight *= DEFAULT_CONFIG['surge_weight']
        elif labels['is_ramp']:
            weight *= DEFAULT_CONFIG['ramp_weight']
        sample_weights[i] = weight
    
    # 分析毛刺模式
    surge_analysis = analyze_surge_patterns(power, surge_extractor=extractor)
    surge_analysis['sample_stats'] = {
        'total_samples': len(X),
        'surge_samples': int(np.sum([l['is_surge'] for l in surge_labels_list])),
        'ramp_samples': int(np.sum([l['is_ramp'] for l in surge_labels_list])),
        'normal_samples': len(X) - int(np.sum([l['is_ramp'] for l in surge_labels_list])),
        'avg_sample_weight': float(sample_weights.mean()),
    }
    
    result = {
        'X': X,
        'y': y,
        'power_history': power_history,
        'surge_features': surge_features,
        'sample_weights': sample_weights,
        'surge_labels': surge_labels_list,
        'feature_cols': feature_cols,
        'config': {
            'lookback': lookback,
            'horizon': horizon,
            'n_features': X.shape[-1],
            'n_surge_features': 3,
        },
    }
    
    return result, surge_analysis


def prepare_surge_aware_samples(
    source_csv: str,
    output_dir: str,
    power_col: str = 'power',
    time_col: str = 'timestamp',
    lookback: int = 16,
    horizon: int = 1,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    surge_threshold_ratio: float = 0.15,
    exclude_night: bool = True,
    min_daylight_power: float = 5.0,
) -> Dict:
    """
    准备完整的毛刺感知样本数据集
    
    Args:
        source_csv: 源 CSV 文件路径
        output_dir: 输出目录
        power_col: 功率列名
        time_col: 时间列名
        lookback: 回看步长
        horizon: 预测步长
        train_frac: 训练集比例
        val_frac: 验证集比例
        test_frac: 测试集比例
        surge_threshold_ratio: 毛刺检测阈值
        exclude_night: 是否排除夜间数据
        min_daylight_power: 日间最小功率阈值
        
    Returns:
        处理结果统计
    """
    print("=" * 60)
    print("毛刺感知样本准备")
    print("=" * 60)
    
    # 读取数据
    print(f"\n读取数据: {source_csv}")
    df = pd.read_csv(source_csv)
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col).reset_index(drop=True)
    
    print(f"原始数据量: {len(df)}")
    
    # 排除夜间（可选）
    if exclude_night:
        # 假设有 GHI (Global Horizontal Irradiance) 列
        if 'ghi' in df.columns:
            df = df[df['ghi'] > min_daylight_power].reset_index(drop=True)
            print(f"排除夜间后: {len(df)}")
    
    # 准备特征
    print(f"\n准备样本: lookback={lookback}, horizon={horizon}")
    
    # 确定特征列
    exclude_cols = [power_col, time_col, 'Unnamed: 0']
    feature_cols = [col for col in df.columns 
                  if col not in exclude_cols and df[col].dtype in [np.float64, np.int64]]
    
    result, surge_analysis = prepare_surge_features(
        df,
        power_col=power_col,
        time_col=time_col,
        feature_cols=feature_cols,
        lookback=lookback,
        horizon=horizon,
        surge_threshold_ratio=surge_threshold_ratio,
    )
    
    print(f"\n毛刺分析结果:")
    print(f"  毛刺样本比例: {surge_analysis['surge_ratio']*100:.1f}%")
    print(f"  方向分布: {surge_analysis['surge_direction_dist']}")
    
    # 划分数据集
    n_samples = len(result['X'])
    n_train = int(n_samples * train_frac)
    n_val = int(n_samples * val_frac)
    
    train_idx = np.arange(0, n_train)
    val_idx = np.arange(n_train, n_train + n_val)
    test_idx = np.arange(n_train + n_val, n_samples)
    
    print(f"\n数据集划分:")
    print(f"  训练: {len(train_idx)} 样本")
    print(f"  验证: {len(val_idx)} 样本")
    print(f"  测试: {len(test_idx)} 样本")
    
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存数据
    print(f"\n保存到: {output_dir}")
    
    # 训练集
    np.save(output_dir / "X_train_seq.npy", result['X'][train_idx])
    np.save(output_dir / "y_train.npy", result['y'][train_idx])
    np.save(output_dir / "power_history_train.npy", result['power_history'][train_idx])
    np.save(output_dir / "surge_features_train.npy", result['surge_features'][train_idx])
    np.save(output_dir / "sample_weights_train.npy", result['sample_weights'][train_idx])
    
    # 验证集
    np.save(output_dir / "X_val_seq.npy", result['X'][val_idx])
    np.save(output_dir / "y_val.npy", result['y'][val_idx])
    np.save(output_dir / "power_history_val.npy", result['power_history'][val_idx])
    np.save(output_dir / "surge_features_val.npy", result['surge_features'][val_idx])
    np.save(output_dir / "sample_weights_val.npy", result['sample_weights'][val_idx])
    
    # 测试集
    np.save(output_dir / "X_test_seq.npy", result['X'][test_idx])
    np.save(output_dir / "y_test.npy", result['y'][test_idx])
    np.save(output_dir / "power_history_test.npy", result['power_history'][test_idx])
    np.save(output_dir / "surge_features_test.npy", result['surge_features'][test_idx])
    np.save(output_dir / "sample_weights_test.npy", result['sample_weights'][test_idx])
    
    # 保存元数据
    meta = {
        'lookback': lookback,
        'horizon': horizon,
        'n_features': result['config']['n_features'],
        'n_surge_features': result['config']['n_surge_features'],
        'feature_cols': result['feature_cols'],
        'surge_threshold_ratio': surge_threshold_ratio,
        'train_samples': len(train_idx),
        'val_samples': len(val_idx),
        'test_samples': len(test_idx),
        'surge_analysis': surge_analysis,
    }
    
    with open(output_dir / "meta.json", 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    
    # 保存毛刺标签
    surge_labels = {
        'train': [result['surge_labels'][i] for i in train_idx],
        'val': [result['surge_labels'][i] for i in val_idx],
        'test': [result['surge_labels'][i] for i in test_idx],
    }
    
    # 转换为可 JSON 序列化的格式
    for split in surge_labels:
        for i in range(len(surge_labels[split])):
            for key, val in surge_labels[split][i].items():
                if isinstance(val, np.ndarray):
                    surge_labels[split][i][key] = val.tolist()
                elif isinstance(val, (np.bool_, np.integer, np.floating)):
                    surge_labels[split][i][key] = bool(val) if isinstance(val, np.bool_) else float(val)
    
    with open(output_dir / "surge_labels.json", 'w', encoding='utf-8') as f:
        json.dump(surge_labels, f, indent=2, ensure_ascii=False)
    
    print("\n文件已保存:")
    for f in output_dir.glob("*.npy"):
        print(f"  {f.name}: {f.stat().st_size / 1024:.1f} KB")
    
    print("\n" + "=" * 60)
    print("样本准备完成！")
    print("=" * 60)
    
    return {
        'output_dir': str(output_dir),
        'meta': meta,
        'surge_analysis': surge_analysis,
    }


def augment_existing_samples(
    sample_dir: str,
    output_dir: Optional[str] = None,
    surge_threshold_ratio: float = 0.15,
    power_col_idx: int = -1,
) -> Dict:
    """
    为现有样本添加毛刺特征
    
    Args:
        sample_dir: 现有样本目录
        output_dir: 输出目录（默认为 sample_dir）
        surge_threshold_ratio: 毛刺检测阈值
        power_col_idx: 功率列索引（默认为最后一列）
        
    Returns:
        处理结果
    """
    sample_dir = Path(sample_dir)
    output_dir = Path(output_dir) if output_dir else sample_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("为现有样本添加毛刺特征")
    print("=" * 60)
    
    # 加载现有数据
    X_train = np.load(sample_dir / "X_train_seq.npy")
    y_train = np.load(sample_dir / "y_train.npy")
    X_val = np.load(sample_dir / "X_val_seq.npy")
    y_val = np.load(sample_dir / "y_val.npy")
    
    print(f"\n训练集: {X_train.shape}")
    print(f"验证集: {X_val.shape}")
    
    # 提取历史功率
    power_train = X_train[:, :, power_col_idx]
    power_val = X_val[:, :, power_col_idx]
    
    # 创建毛刺特征
    extractor = SurgeFeatureExtractor(surge_threshold_ratio=surge_threshold_ratio)
    
    print("\n计算毛刺特征...")
    
    # 训练集
    surge_features_train = []
    for i in range(len(X_train)):
        features = extractor.extract_surge_features(
            power_train[i].reshape(1, -1),
            power_future=y_train[i].reshape(1, -1)
        )
        surge_dir = features['surge_direction'].squeeze()
        surge_mag = features['surge_magnitude'].squeeze()
        surge_mask = features['surge_mask'].squeeze()
        
        surge_features_train.append(np.stack([
            surge_dir,
            surge_mag / (surge_mag.max() + 1e-8),
            surge_mask.astype(float),
        ], axis=-1))
    
    surge_features_train = np.stack(surge_features_train, axis=0)
    
    # 验证集
    surge_features_val = []
    for i in range(len(X_val)):
        features = extractor.extract_surge_features(
            power_val[i].reshape(1, -1),
            power_future=y_val[i].reshape(1, -1)
        )
        surge_dir = features['surge_direction'].squeeze()
        surge_mag = features['surge_magnitude'].squeeze()
        surge_mask = features['surge_mask'].squeeze()
        
        surge_features_val.append(np.stack([
            surge_dir,
            surge_mag / (surge_mag.max() + 1e-8),
            surge_mask.astype(float),
        ], axis=-1))
    
    surge_features_val = np.stack(surge_features_val, axis=0)
    
    # 保存
    np.save(output_dir / "surge_features_train.npy", surge_features_train)
    np.save(output_dir / "surge_features_val.npy", surge_features_val)
    
    # 分析
    surge_analysis = {
        'train': analyze_surge_patterns(power_train, surge_extractor=extractor),
        'val': analyze_surge_patterns(power_val, surge_extractor=extractor),
    }
    
    print(f"\n训练集毛刺分析: {surge_analysis['train']['surge_ratio']*100:.1f}% 样本有毛刺")
    print(f"验证集毛刺分析: {surge_analysis['val']['surge_ratio']*100:.1f}% 样本有毛刺")
    
    # 更新 meta.json
    meta_path = sample_dir / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        meta['n_surge_features'] = 3
        meta['surge_threshold_ratio'] = surge_threshold_ratio
        meta['surge_analysis'] = surge_analysis
        
        with open(output_dir / "meta.json", 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
    
    print(f"\n毛刺特征已保存到: {output_dir}")
    
    return {
        'output_dir': str(output_dir),
        'surge_analysis': surge_analysis,
    }


# ============================================================================
# 命令行入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="毛刺感知样本准备")
    
    parser.add_argument("--mode", type=str, required=True,
                        choices=['prepare', 'augment'],
                        help="模式: prepare=从头准备, augment=增强现有样本")
    
    parser.add_argument("--source", type=str,
                        help="源CSV文件（prepare模式需要）")
    parser.add_argument("--sample-dir", type=str,
                        help="样本目录（augment模式需要）")
    parser.add_argument("--output", type=str,
                        help="输出目录")
    
    parser.add_argument("--power-col", type=str, default="power",
                        help="功率列名")
    parser.add_argument("--time-col", type=str, default="timestamp",
                        help="时间列名")
    
    parser.add_argument("--lookback", type=int, default=16,
                        help="回看步长")
    parser.add_argument("--horizon", type=int, default=1,
                        help="预测步长")
    
    parser.add_argument("--surge-threshold", type=float, default=0.15,
                        help="毛刺检测阈值")
    
    parser.add_argument("--train-frac", type=float, default=0.7,
                        help="训练集比例")
    parser.add_argument("--val-frac", type=float, default=0.15,
                        help="验证集比例")
    
    parser.add_argument("--exclude-night", action="store_true",
                        help="排除夜间数据")
    
    args = parser.parse_args()
    
    if args.mode == 'prepare':
        if not args.source or not args.output:
            parser.error("prepare模式需要 --source 和 --output")
        
        result = prepare_surge_aware_samples(
            source_csv=args.source,
            output_dir=args.output,
            power_col=args.power_col,
            time_col=args.time_col,
            lookback=args.lookback,
            horizon=args.horizon,
            train_frac=args.train_frac,
            val_frac=args.val_frac,
            surge_threshold_ratio=args.surge_threshold,
            exclude_night=args.exclude_night,
        )
        
    elif args.mode == 'augment':
        if not args.sample_dir:
            parser.error("augment模式需要 --sample-dir")
        
        output_dir = args.output or args.sample_dir
        
        result = augment_existing_samples(
            sample_dir=args.sample_dir,
            output_dir=output_dir,
            surge_threshold_ratio=args.surge_threshold,
        )
    
    return result


if __name__ == "__main__":
    main()
