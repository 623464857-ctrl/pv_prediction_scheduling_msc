"""
亳州数据集样本准备脚本 (EXP-P05)
生成超参数搜索所需的序列样本数据

python -m experiments.prediction.step2_hyperparameter_search.exp_02_prepare_samples_bozhou
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.prediction.step2_hyperparameter_search.exp_02_common import (
    setup_logger,
    STEP4_ROOT,
)


# =============================================================================
# 特征工程配置
# =============================================================================

# 输入特征列表 (与 meta.json 一致)
INPUT_FEATURES = [
    "power_pu_lag_0",
    "power_pu_lag_1",
    "power_pu_lag_2",
    "power_pu_lag_4",
    "power_pu_lag_8",
    "power_pu_lag_16",
    "power_ramp_15m_pu",
    "power_ramp_60m_pu",
    "power_ramp_120m_pu",
    "power_pu_roll_1h_mean",
    "power_pu_roll_1h_std",
    "power_pu_roll_2h_mean",
    "power_pu_roll_2h_std",
    "power_pu_roll_2h_max",
    "power_pu_roll_2h_min",
    "ghi_wm2",
    "temperature_c",
    "relative_humidity_pct",
    "cloud_cover_pct",
    "visibility_km",
    "sin_hour",
    "cos_hour",
    "is_daytime",
    "data_quality_score",
]


def create_sequences(df: pd.DataFrame, lookback: int, horizon: int, 
                     feature_cols: list[str], target_col: str = "power_pu") -> tuple:
    """创建序列样本

    Args:
        df: 预处理后的数据
        lookback: 回看步数
        horizon: 预测步数
        feature_cols: 特征列名
        target_col: 目标列名

    Returns:
        X: 特征序列 (n_samples, lookback, n_features)
        y: 目标值 (n_samples, horizon)
        timestamps: 对应的时间戳
    """
    X, y, timestamps = [], [], []
    
    # 特征索引
    feature_idx = [df.columns.get_loc(col) for col in feature_cols if col in df.columns]
    target_idx = df.columns.get_loc(target_col)
    
    n = len(df)
    max_start = n - lookback - horizon + 1
    
    for i in range(max_start):
        # 特征序列: 从 i 到 i + lookback - 1
        x_seq = df.iloc[i : i + lookback, feature_idx].values.astype(np.float32)
        X.append(x_seq)
        
        # 目标: 从 i + lookback 到 i + lookback + horizon - 1
        y_target = df.iloc[i + lookback : i + lookback + horizon, target_idx].values.astype(np.float32)
        y.append(y_target)
        
        # 时间戳: 使用预测起点的时间戳
        timestamps.append(df.iloc[i + lookback, df.columns.get_loc("timestamp")])
    
    return np.array(X), np.array(y), np.array(timestamps)


def prepare_bozhou_samples(
    data_file: Path,
    horizon: int,
    lookback: int,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    capacity_kw: float = 220.0,
) -> dict:
    """准备亳州数据集的样本

    Args:
        data_file: 预处理后的数据文件路径
        horizon: 预测步数
        lookback: 回看步数
        train_frac: 训练集比例
        val_frac: 验证集比例
        test_frac: 测试集比例
        capacity_kw: 装机容量 (kW)

    Returns:
        样本统计信息
    """
    print(f"\n{'='*60}")
    print(f"准备亳州数据集样本: horizon={horizon}, lookback={lookback}")
    print(f"{'='*60}")
    
    # 1. 加载数据
    print("\n[1] 加载数据...")
    df = pd.read_csv(data_file)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    print(f"  数据行数: {len(df):,}")
    
    # 2. 创建衍生特征
    print("\n[2] 创建特征...")
    
    # 计算 lag 特征 (如果不存在)
    if "power_pu_lag_0" not in df.columns:
        print("  创建 power_pu lag 特征...")
        df["power_pu"] = df["power_kw"] / capacity_kw
        df["power_pu_lag_0"] = df["power_pu"].shift(0)
        df["power_pu_lag_1"] = df["power_pu"].shift(1)
        df["power_pu_lag_2"] = df["power_pu"].shift(2)
        df["power_pu_lag_4"] = df["power_pu"].shift(4)
        df["power_pu_lag_8"] = df["power_pu"].shift(8)
        df["power_pu_lag_16"] = df["power_pu"].shift(16)
    
    # 计算 ramp 特征 (如果不存在)
    if "power_ramp_60m_pu" not in df.columns:
        print("  创建 power_ramp 特征...")
        df["power_ramp_60m_pu"] = df["power_pu"].diff(4)  # 60min = 4 * 15min
        df["power_ramp_120m_pu"] = df["power_pu"].diff(8)  # 120min = 8 * 15min
    
    # 计算滚动统计特征 (如果不存在)
    if "power_pu_roll_1h_mean" not in df.columns:
        print("  创建 rolling 特征...")
        df["power_pu_roll_1h_mean"] = df["power_pu"].rolling(4, min_periods=1).mean()
        df["power_pu_roll_1h_std"] = df["power_pu"].rolling(4, min_periods=1).std()
        df["power_pu_roll_2h_mean"] = df["power_pu"].rolling(8, min_periods=1).mean()
        df["power_pu_roll_2h_std"] = df["power_pu"].rolling(8, min_periods=1).std()
        df["power_pu_roll_2h_max"] = df["power_pu"].rolling(8, min_periods=1).max()
        df["power_pu_roll_2h_min"] = df["power_pu"].rolling(8, min_periods=1).min()
    
    # 填充 NaN
    df = df.fillna(0)
    
    # 3. 检查特征是否存在
    missing_features = [f for f in INPUT_FEATURES if f not in df.columns]
    if missing_features:
        print(f"  [!] 缺失特征: {missing_features}")
        # 只使用存在的特征
        feature_cols = [f for f in INPUT_FEATURES if f in df.columns]
    else:
        feature_cols = INPUT_FEATURES
    
    print(f"  使用 {len(feature_cols)} 个特征")
    
    # 4. 创建序列样本
    print("\n[3] 创建序列样本...")
    X, y, timestamps = create_sequences(df, lookback, horizon, feature_cols)
    print(f"  样本数: {len(X):,}")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    
    # 5. 划分数据集
    print("\n[4] 划分数据集...")
    n_total = len(X)
    n_train = int(n_total * train_frac)
    n_val = int(n_total * val_frac)
    # n_test = n_total - n_train - n_val
    
    train_end = n_train
    val_end = n_train + n_val
    
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    
    print(f"  训练集: {len(X_train):,} ({len(X_train)/n_total*100:.1f}%)")
    print(f"  验证集: {len(X_val):,} ({len(X_val)/n_total*100:.1f}%)")
    print(f"  测试集: {len(X_test):,} ({len(X_test)/n_total*100:.1f}%)")
    
    # 6. 保存样本
    print("\n[5] 保存样本...")
    sample_dir = STEP4_ROOT / "samples" / f"bozhou_h{horizon}_lb{lookback}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存 numpy 数组
    np.save(sample_dir / "X_train_seq.npy", X_train)
    np.save(sample_dir / "y_train.npy", y_train)
    np.save(sample_dir / "X_val_seq.npy", X_val)
    np.save(sample_dir / "y_val.npy", y_val)
    np.save(sample_dir / "X_test_seq.npy", X_test)
    np.save(sample_dir / "y_test.npy", y_test)
    
    # 保存测试集时间戳
    test_timestamps = pd.DataFrame({
        "timestamp": timestamps[val_end:],
        "horizon": horizon,
    })
    test_timestamps.to_csv(sample_dir / "test_timestamps.csv", index=False)
    
    # 7. 保存元信息
    meta = {
        "dataset": "bozhou",
        "capacity_kw": capacity_kw,
        "capacity_mw": capacity_kw / 1000,
        "lookback": lookback,
        "horizon": horizon,
        "n_features": len(feature_cols),
        "feature_cols": feature_cols,
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
        "train_frac": train_frac,
        "val_frac": val_frac,
        "test_frac": test_frac,
        "source_csv": str(data_file),
        "total_windows": n_total,
        "prediction_mode": "direct",
    }
    with open(sample_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    
    # 8. 保存 scaler 参数 (用于反归一化)
    # 对 y 进行标准化
    y_scaler = StandardScaler()
    y_scaler.fit(y_train)
    
    scaler_params = {
        "y_mean": y_scaler.mean_.tolist(),
        "y_scale": y_scaler.scale_.tolist(),
    }
    with open(sample_dir / "scaler_params.json", "w", encoding="utf-8") as f:
        json.dump(scaler_params, f, indent=2, ensure_ascii=False)
    
    print(f"\n  样本已保存到: {sample_dir}")
    
    return meta


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="亳州数据集样本准备")
    parser.add_argument("--horizon", type=int, nargs="+", default=[1, 4, 16],
                        help="预测步数 (默认: 1 4 16)")
    parser.add_argument("--lookback", type=int, nargs="+", default=None,
                        help="回看步数 (默认: 根据 horizon 自动设置)")
    parser.add_argument("--data", type=str, default=None,
                        help="输入数据文件路径")
    args = parser.parse_args()
    
    # 默认数据文件
    if args.data is None:
        data_file = PROJECT_ROOT / "data/prediction/step1_preprocessing/processed/stations/Bozhou_1_cleaned.csv"
    else:
        data_file = Path(args.data)
    
    if not data_file.exists():
        print(f"错误: 文件不存在 {data_file}")
        # 尝试备用文件
        data_file = PROJECT_ROOT / "data/prediction/step1_preprocessing/processed/stations/Bozhou_1_preprocessed.csv"
        if data_file.exists():
            print(f"  使用备用文件: {data_file}")
        else:
            return
    
    # 回看步数映射
    lookback_map = {1: 16, 4: 48, 16: 96}
    
    # 设置日志
    log_file = "EXP-P05_bozhou_prepare_samples.log"
    logger = setup_logger("prepare_bozhou", log_file)
    logger.info("=" * 60)
    logger.info("亳州数据集样本准备")
    logger.info("输入数据: %s", data_file)
    
    t0 = time.time()
    results = {}
    
    for horizon in args.horizon:
        lookback = args.lookback[0] if args.lookback else lookback_map.get(horizon, 16)
        
        try:
            meta = prepare_bozhou_samples(
                data_file=data_file,
                horizon=horizon,
                lookback=lookback,
                train_frac=0.70,
                val_frac=0.15,
                test_frac=0.15,
                capacity_kw=220.0,
            )
            results[horizon] = meta
            logger.info("horizon=%d 完成: train=%d val=%d test=%d",
                       horizon, meta["n_train"], meta["n_val"], meta["n_test"])
        except Exception as e:
            logger.error("horizon=%d 失败: %s", horizon, str(e))
            import traceback
            logger.error(traceback.format_exc())
    
    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info("全部完成，耗时: %.1f 秒", elapsed)
    
    print(f"\n{'='*60}")
    print(f"样本准备完成！耗时: {elapsed:.1f} 秒")
    print(f"{'='*60}")
    
    return results


if __name__ == "__main__":
    main()
