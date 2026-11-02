"""
预测层与调度层共用的路径、常量和小工具。

所有脚本都从这里取路径，改目录结构时只需要改这一个文件。
"""

import json
import logging
import random
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# 配置后缀（由 run_4configs.py 动态设置，格式如 "simple_sigma3"）
# 各配置使用独立的子目录避免结果冲突
# ---------------------------------------------------------------------------
_CFG_SUFFIX = "two_stage_sigma3"  # run_4configs


def set_config_suffix(suffix: str) -> None:
    """run_4configs.py 调用此函数设置配置后缀，各子目录自动带上后缀"""
    global _CFG_SUFFIX
    _CFG_SUFFIX = suffix


def _cfg_subdir(base: Path) -> Path:
    """如果设置了后缀，返回 base/<suffix>，否则返回 base"""
    if _CFG_SUFFIX:
        return base / _CFG_SUFFIX
    return base


# ---------------------------------------------------------------------------
# 数据
# ---------------------------------------------------------------------------
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "亳州"
RAW_POWER_CSV = RAW_DIR / "亳州_15min.csv"                    # 光伏功率 (15 min)
RAW_WEATHER_CSV = RAW_DIR / "亳州3-8月天气数据.csv"            # Weatherbit 天气 (15 min)
SCHEDULE_RAW_DIR = PROJECT_ROOT / "data" / "schedule_data"   # 光储充运行数据 (5 min, xlsx)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

def _processed_csv(filename_base: str) -> Path:
    """生成配置特定的 processed 目录下的 CSV 路径"""
    if _CFG_SUFFIX:
        return PROCESSED_DIR / f"{filename_base}_{_CFG_SUFFIX}.csv"
    return PROCESSED_DIR / f"{filename_base}.csv"

PREPROCESSED_CSV = _processed_csv("Bozhou_1_preprocessed")  # p1 输出
CLEANED_CSV = _processed_csv("Bozhou_1_cleaned")           # p2 输出 (预测模型的输入数据)
QUALITY_REPORT_DIR = PROCESSED_DIR / "data_quality"
SCHEDULE_CSV = PROCESSED_DIR / "schedule_15min.csv"       # s1 输出 (调度模型的输入数据)

# ---------------------------------------------------------------------------
# 结果
# ---------------------------------------------------------------------------
PRED_RESULT_DIR = PROJECT_ROOT / "result" / "prediction"
SAMPLES_DIR = _cfg_subdir(PRED_RESULT_DIR / "samples")   # 滑动窗口样本
MODELS_DIR = _cfg_subdir(PRED_RESULT_DIR / "models")    # 训练好的模型
PARAMS_DIR = _cfg_subdir(PRED_RESULT_DIR / "params")    # 超参数 (搜索结果)
PREDICTIONS_DIR = _cfg_subdir(PRED_RESULT_DIR / "predictions")  # 测试集预测
PRED_FIGURES_DIR = _cfg_subdir(PRED_RESULT_DIR / "figures")

SCHED_RESULT_DIR = PROJECT_ROOT / "result" / "scheduling"

LOG_DIR = PROJECT_ROOT / "logs"

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
CAPACITY_KW = 390.0                    # 功率标幺化基准 power_pu = power_kw / CAPACITY_KW
LOOKBACK = {1: 16, 4: 48, 16: 96}      # 预测步长 H -> 回看步数 L (15 min/步)

# 默认超参数 (没有搜索结果时使用)
DEFAULT_PARAMS = {
    "lr": 0.002, "batch_size": 64,
    "hidden_size": 128, "num_layers": 2, "dropout": 0.1,
    "cnn_channels": [32, 64], "kernel_size": 5,
}


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def setup_logger(name: str, log_file: str) -> logging.Logger:
    """同时输出到终端和 logs/<log_file> (每次运行覆盖旧日志)。"""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("[%(levelname)s] %(message)s")
    for handler in (logging.StreamHandler(), logging.FileHandler(LOG_DIR / log_file, mode="w", encoding="utf-8")):
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


def set_seed(seed: int) -> None:
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sample_dir(horizon: int) -> Path:
    """配置特定的样本目录，如 result/prediction/samples/simple_sigma3/h1_lb16/"""
    base = PRED_RESULT_DIR / "samples"
    if _CFG_SUFFIX:
        base = base / _CFG_SUFFIX
    return base / f"h{horizon}_lb{LOOKBACK[horizon]}"


def params_path(horizon: int) -> Path:
    """配置特定的超参数文件路径"""
    base = PRED_RESULT_DIR / "params"
    if _CFG_SUFFIX:
        base = base / _CFG_SUFFIX
    return base / f"best_params_h{horizon}.json"


def model_path(horizon: int, mode: str, seed: int) -> Path:
    """配置特定的模型文件路径"""
    base = PRED_RESULT_DIR / "models"
    if _CFG_SUFFIX:
        base = base / _CFG_SUFFIX
    return base / f"h{horizon}_{mode}_seed{seed}.pt"


def load_params(horizon: int) -> dict:
    """读取超参数: 优先用 p4 搜索得到的结果, 否则用 DEFAULT_PARAMS。"""
    path = params_path(horizon)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))["params"]
    return dict(DEFAULT_PARAMS)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """MAE / RMSE / MAPE / R2。MAPE 只在 |y_true| > 0.01 的点上计算, 避免夜间除零。"""
    y_true, y_pred = np.ravel(y_true), np.ravel(y_pred)
    err = y_pred - y_true
    mask = np.abs(y_true) > 0.01
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "MAPE": float(np.mean(np.abs(err[mask] / y_true[mask])) * 100) if mask.any() else float("nan"),
        "R2": float(1 - np.sum(err ** 2) / np.sum((y_true - y_true.mean()) ** 2)),
    }
