"""
实验编号: EXP-P01-DS
实验名称: 亳州数据集划分
实验目的: 将白昼数据划分为 70%/15%/15% 的 train/val/test 集
所属方向: prediction / step1_preprocessing
输入路径: data/prediction/step1_preprocessing/processed/stations/Bozhou_1_preprocessed.csv
输出路径: data/prediction/step1_preprocessing/splits/
运行方式: python -m experiments.prediction.step1_preprocessing.exp_01_split_data
主要流程: 加载预处理数据 → 筛选白昼数据 → 时间顺序划分 → 保存结果

数据集划分方案:
- 训练集: 70% (按时间顺序)
- 验证集: 15% (按时间顺序)
- 测试集: 15% (按时间顺序)
- 筛选条件: is_daytime == 1
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 路径与常量
# ---------------------------------------------------------------------------
SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[3]
INPUT_PATH = PROJECT_ROOT / "data" / "prediction" / "step1_preprocessing" / "processed" / "stations" / "Bozhou_1_preprocessed.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "prediction" / "step1_preprocessing" / "splits"
LOG_DIR = PROJECT_ROOT / "logs" / "prediction" / "step1_preprocessing"

# 划分比例
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# 白昼筛选条件
DAY_FILTER = "is_daytime == 1"

# 日志配置
LOG_FILE = LOG_DIR / "EXP-P01-DS_split_data.log"


def setup_logging() -> logging.Logger:
    """配置日志"""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger("EXP-P01-DS")
    logger.setLevel(logging.INFO)
    
    # 文件处理器
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setLevel(logging.INFO)
    
    # 控制台处理器
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


def load_data(logger: logging.Logger) -> pd.DataFrame:
    """加载预处理数据"""
    logger.info("正在加载数据: %s", INPUT_PATH)
    
    df = pd.read_csv(INPUT_PATH, parse_dates=["timestamp"])
    logger.info("原始数据: %d 行, %d 列", len(df), len(df.columns))
    
    return df


def filter_daytime(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """筛选白昼数据"""
    df_day = df[df["is_daytime"] == 1].copy()
    logger.info("白昼数据筛选: %d 行 (原始 %d 行, 占比 %.2f%%)", 
                 len(df_day), len(df), len(df_day) / len(df) * 100)
    return df_day


def split_data(
    df: pd.DataFrame, 
    train_ratio: float, 
    val_ratio: float, 
    test_ratio: float,
    logger: logging.Logger
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    按时间顺序划分数据集
    
    参数:
        df: 预处理后的数据 (已按时间排序)
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
    
    返回:
        (train_df, val_df, test_df)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例之和必须等于1"
    
    n = len(df)
    
    # 计算划分点
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    # 验证划分
    assert len(train_df) + len(val_df) + len(test_df) == n
    assert len(train_df) == train_end
    assert len(val_df) == val_end - train_end
    assert len(test_df) == n - val_end
    
    logger.info("=" * 60)
    logger.info("数据集划分结果 (仅白昼数据)")
    logger.info("=" * 60)
    logger.info("训练集: %d 行 (%.2f%%) [%s ~ %s]", 
                len(train_df), train_ratio * 100,
                train_df["timestamp"].min(), train_df["timestamp"].max())
    logger.info("验证集: %d 行 (%.2f%%) [%s ~ %s]", 
                len(val_df), val_ratio * 100,
                val_df["timestamp"].min(), val_df["timestamp"].max())
    logger.info("测试集: %d 行 (%.2f%%) [%s ~ %s]", 
                len(test_df), test_ratio * 100,
                test_df["timestamp"].min(), test_df["timestamp"].max())
    logger.info("=" * 60)
    
    return train_df, val_df, test_df


def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path,
    logger: logging.Logger
) -> dict:
    """保存划分结果"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存 CSV 文件
    train_path = output_dir / "train.csv"
    val_path = output_dir / "val.csv"
    test_path = output_dir / "test.csv"
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    logger.info("已保存: %s", train_path)
    logger.info("已保存: %s", val_path)
    logger.info("已保存: %s", test_path)
    
    # 生成并保存元信息
    meta = {
        "split_date": datetime.now().isoformat(),
        "filter": {"is_daytime": 1},
        "ratios": {
            "train": TRAIN_RATIO,
            "val": VAL_RATIO,
            "test": TEST_RATIO
        },
        "counts": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df)
        },
        "time_ranges": {
            "train": {
                "start": str(train_df["timestamp"].min()),
                "end": str(train_df["timestamp"].max())
            },
            "val": {
                "start": str(val_df["timestamp"].min()),
                "end": str(val_df["timestamp"].max())
            },
            "test": {
                "start": str(test_df["timestamp"].min()),
                "end": str(test_df["timestamp"].max())
            }
        },
        "features": {
            "total": len(train_df.columns),
            "list": list(train_df.columns)
        }
    }
    
    meta_path = output_dir / "split_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    
    logger.info("已保存: %s", meta_path)
    
    return meta


def print_summary(meta: dict, logger: logging.Logger):
    """打印最终摘要"""
    logger.info("")
    logger.info("=" * 60)
    logger.info("数据集划分摘要")
    logger.info("=" * 60)
    logger.info("筛选条件: is_daytime == 1 (仅白昼数据)")
    logger.info("划分比例: Train %.0f%% / Val %.0f%% / Test %.0f%%", 
                meta["ratios"]["train"] * 100,
                meta["ratios"]["val"] * 100,
                meta["ratios"]["test"] * 100)
    logger.info("")
    logger.info("样本数量:")
    logger.info("  训练集: %d 样本", meta["counts"]["train"])
    logger.info("  验证集: %d 样本", meta["counts"]["val"])
    logger.info("  测试集: %d 样本", meta["counts"]["test"])
    logger.info("  总计:   %d 样本", 
                meta["counts"]["train"] + meta["counts"]["val"] + meta["counts"]["test"])
    logger.info("")
    logger.info("时间范围:")
    logger.info("  训练集: %s ~ %s", 
                meta["time_ranges"]["train"]["start"],
                meta["time_ranges"]["train"]["end"])
    logger.info("  验证集: %s ~ %s", 
                meta["time_ranges"]["val"]["start"],
                meta["time_ranges"]["val"]["end"])
    logger.info("  测试集: %s ~ %s", 
                meta["time_ranges"]["test"]["start"],
                meta["time_ranges"]["test"]["end"])
    logger.info("")
    logger.info("输出目录: %s", OUTPUT_DIR)
    logger.info("=" * 60)


def main():
    """主函数"""
    logger = setup_logging()
    logger.info("EXP-P01-DS 数据集划分开始")
    logger.info("时间: %s", datetime.now().isoformat())
    logger.info("")
    
    # 1. 加载数据
    df = load_data(logger)
    
    # 2. 筛选白昼数据
    df_day = filter_daytime(df, logger)
    
    # 3. 数据划分
    train_df, val_df, test_df = split_data(
        df_day, 
        TRAIN_RATIO, 
        VAL_RATIO, 
        TEST_RATIO,
        logger
    )
    
    # 4. 保存结果
    meta = save_splits(train_df, val_df, test_df, OUTPUT_DIR, logger)
    
    # 5. 打印摘要
    print_summary(meta, logger)
    
    logger.info("")
    logger.info("EXP-P01-DS 数据集划分完成")


if __name__ == "__main__":
    main()
