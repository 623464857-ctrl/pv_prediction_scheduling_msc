"""
实验编号: EXP-P03-AFSA-PatchTST
实验名称: 混合深度学习模型对比 — AFSA-PatchTST
实验目的: 使用人工鱼群算法搜索 PatchTST 超参数并训练最优模型
运行方式: python experiments/prediction/step3_hybrid_models/run_exp_p03_afsa_patchtst.py
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from exp_p03_afsa import main as afsa_main

LOG_NAME = "EXP-P03_AFSA_PatchTST.log"


def main() -> None:
    afsa_main()


if __name__ == "__main__":
    main()
