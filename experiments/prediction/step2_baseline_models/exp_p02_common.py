"""EXP-P02 共享路径、配置与日志工具。"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
STEP2_ROOT = PROJECT_ROOT / "data" / "prediction" / "step2_baseline_models"
CONFIG_PATH = STEP2_ROOT / "config" / "exp_p02_config.json"
SAMPLES_DIR = STEP2_ROOT / "samples"
MODELS_DIR = STEP2_ROOT / "models"
PRED_DIR = STEP2_ROOT / "predictions"
METRICS_DIR = STEP2_ROOT / "metrics"
FIGURES_DIR = STEP2_ROOT / "figures"
REPORTS_DIR = STEP2_ROOT / "reports"
LOG_DIR = PROJECT_ROOT / "logs" / "prediction" / "step2_baseline_models"


def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


def setup_logger(name: str, log_file: str) -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / log_file
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8", mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    logger = logging.getLogger(name)
    logger.info("日志文件: %s", log_path.relative_to(PROJECT_ROOT))
    return logger


def append_log_summary(log_file: str, lines: list[str]) -> None:
    log_path = LOG_DIR / log_file
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n" + "\n".join(lines) + "\n")


def ensure_dirs() -> None:
    for d in (MODELS_DIR, PRED_DIR, METRICS_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_samples() -> dict:
    import numpy as np

    return {
        "X_train_flat": np.load(SAMPLES_DIR / "X_train_flat.npy"),
        "X_val_flat": np.load(SAMPLES_DIR / "X_val_flat.npy"),
        "X_test_flat": np.load(SAMPLES_DIR / "X_test_flat.npy"),
        "X_train_seq": np.load(SAMPLES_DIR / "X_train_seq.npy"),
        "X_val_seq": np.load(SAMPLES_DIR / "X_val_seq.npy"),
        "X_test_seq": np.load(SAMPLES_DIR / "X_test_seq.npy"),
        "y_train": np.load(SAMPLES_DIR / "y_train.npy"),
        "y_val": np.load(SAMPLES_DIR / "y_val.npy"),
        "y_test": np.load(SAMPLES_DIR / "y_test.npy"),
    }


def load_test_timestamps():
    import pandas as pd

    return pd.read_csv(SAMPLES_DIR / "test_timestamps.csv", parse_dates=["timestamp"])["timestamp"]


def save_predictions(model_name: str, y_true, y_pred) -> Path:
    import pandas as pd

    ts = load_test_timestamps()
    out = pd.DataFrame(
        {
            "timestamp": ts.values,
            "y_true": y_true,
            "y_pred": y_pred,
            "model_name": model_name,
        }
    )
    path = PRED_DIR / f"{model_name.lower()}_test.csv"
    out.to_csv(path, index=False)
    return path


def save_train_history(model_key: str, history: list[dict]) -> Path:
    import pandas as pd

    path = METRICS_DIR / f"{model_key}_train_history.csv"
    pd.DataFrame(history).to_csv(path, index=False)
    return path
