"""
2配置对比实验运行器

生成2个配置文件（单阶段Hampel方法 × 2种Sigma阈值），分别运行完整预处理→预测流程，并输出对比结果。

配置:
  1. simple_sigma3  - 单阶段Hampel, 3σ
  2. simple_sigma6  - 单阶段Hampel, 6σ

python -m experiments.prediction.run_4configs
"""

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
COMMON_PY = PROJECT_ROOT / "experiments" / "common.py"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PRED_RESULT_DIR = PROJECT_ROOT / "result" / "prediction"

# 2个配置 (two_stage方法)
CONFIGS = [
    {"name": "two_stage_sigma3", "method": "two_stage", "sigma": 3.0},
    {"name": "two_stage_sigma6", "method": "two_stage", "sigma": 6.0},
]

# 预测步长 (所有H)
DEFAULT_HORIZONS = [1, 4, 16]
DEFAULT_MODE = "optuna"  # 改为纯 Optuna 搜索
DEFAULT_SEED = 42


def log(msg: str):
    """带时间戳的日志输出"""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def run_cmd(cmd: list[str], desc: str, timeout: int = 3600) -> subprocess.CompletedProcess:
    """执行命令并打印结果"""
    log(f"-> {desc}")
    log(f"  命令: {' '.join(str(c) for c in cmd)}")
    t0 = time.time()
    try:
        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            cwd=str(PROJECT_ROOT),
            env=env,
        )
        elapsed = time.time() - t0
        if result.stdout:
            print(result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout, flush=True)
        if result.stderr:
            print(f"  [STDERR] {result.stderr[-2000:]}", flush=True)
        if result.returncode != 0:
            log(f"  X 失败 (exit {result.returncode}, 耗时 {elapsed:.1f}s)")
        else:
            log(f"  OK 完成 (耗时 {elapsed:.1f}s)")
        return result
    except subprocess.TimeoutExpired:
        log(f"  X 超时 ({timeout}s)")
        raise


def set_config_suffix_in_common(cfg_name: str) -> None:
    """在 common.py 中设置 _CFG_SUFFIX，供各脚本在 import 时读取"""
    content = COMMON_PY.read_text(encoding="utf-8")

    # 替换 _CFG_SUFFIX = ""
    content = re.sub(
        r'^_CFG_SUFFIX = .*$',
        f'_CFG_SUFFIX = "{cfg_name}"  # run_4configs',
        content,
        flags=re.MULTILINE,
    )
    COMMON_PY.write_text(content, encoding="utf-8")


def reset_config_suffix() -> None:
    """重置 common.py 中的 _CFG_SUFFIX"""
    content = COMMON_PY.read_text(encoding="utf-8")
    content = re.sub(
        r'^_CFG_SUFFIX = .*$',
        '_CFG_SUFFIX = ""  # run_4configs',
        content,
        flags=re.MULTILINE,
    )
    COMMON_PY.write_text(content, encoding="utf-8")


def collect_metrics(cfg_name: str, horizon: int, mode: str) -> dict:
    """从 p6 的 metrics JSON 中读取结果（使用配置特定的路径）"""
    # 配置特定的 PREDICTIONS_DIR
    predictions_dir = PRED_RESULT_DIR / "predictions" / cfg_name
    metrics_file = predictions_dir / f"h{horizon}_{mode}_metrics.json"
    if not metrics_file.exists():
        # 尝试默认路径（无后缀）
        metrics_file = PRED_RESULT_DIR / "predictions" / f"h{horizon}_{mode}_metrics.json"
    if not metrics_file.exists():
        log(f"  警告: 找不到 metrics 文件 {metrics_file}")
        return {}
    data = json.loads(metrics_file.read_text(encoding="utf-8"))
    test = data.get("test", {})
    model = test.get("model", {})
    persist = test.get("persistence", {})
    return {
        "RMSE": model.get("RMSE"),
        "MAE": model.get("MAE"),
        "MAPE": model.get("MAPE"),
        "R2": model.get("R2"),
        "persist_RMSE": persist.get("RMSE"),
        "skill": (
            round((1 - model.get("RMSE", 0) / persist.get("RMSE", 1)) * 100, 2)
            if persist.get("RMSE", 0) else None
        ),
    }


def main():
    t_start = time.time()
    print("=" * 70)
    print("2配置对比实验")
    print("=" * 70)
    print(f"配置列表:")
    for i, cfg in enumerate(CONFIGS, 1):
        print(f"  {i}. {cfg['name']:20s}  method={cfg['method']:10s}  sigma={cfg['sigma']}")
    print(f"预测步长: H={DEFAULT_HORIZONS}, 模式={DEFAULT_MODE}, seed={DEFAULT_SEED}")
    print("=" * 70)

    results = {}

    for idx, cfg in enumerate(CONFIGS, 1):
        cfg_name = cfg["name"]
        method = cfg["method"]
        sigma = cfg["sigma"]
        elapsed_cfg = time.time()

        print("\n" + "=" * 70)
        print(f"配置 {idx}/4: {cfg_name} (method={method}, sigma={sigma})")
        print("=" * 70)

        # 1. 更新 common.py 中的 _CFG_SUFFIX
        log(f"--- 开始配置 {idx}/2: {cfg_name} (method={method}, sigma={sigma}) ---")
        set_config_suffix_in_common(cfg_name)

        # 2. 运行 p1 预处理
        log(f"Step 1/4: 运行 p1 预处理")
        r = run_cmd(
            [sys.executable, "-m", "experiments.prediction.p1_preprocess",
             "--method", method, "--sigma", str(sigma)],
            f"p1_preprocess [{cfg_name}]",
            timeout=600,
        )
        if r.returncode != 0:
            log(f"X p1 失败，跳过配置 {cfg_name}")
            results[cfg_name] = {"error": "p1 failed"}
            continue

        # 3. 运行 p2 质量检查
        log(f"Step 2/4: 运行 p2 质量检查")
        r = run_cmd(
            [sys.executable, "-m", "experiments.prediction.p2_quality_check", "--mode", "full"],
            f"p2_quality_check [{cfg_name}]",
            timeout=300,
        )
        if r.returncode != 0:
            log(f"X p2 失败，跳过配置 {cfg_name}")
            results[cfg_name] = {"error": "p2 failed"}
            continue

        # 对每个 horizon 分别运行 p3-p6
        for horizon in DEFAULT_HORIZONS:
            log(f"\n--- H={horizon} ---")

            # 4. 运行 p3 样本制备
            log(f"Step 3/4: 运行 p3 样本制备")
            r = run_cmd(
                [sys.executable, "-m", "experiments.prediction.p3_prepare_samples",
                 "--horizon", str(horizon)],
                f"p3_prepare_samples [{cfg_name}] H={horizon}",
                timeout=300,
            )
            if r.returncode != 0:
                log(f"X p3 失败，跳过 H={horizon}")
                results.setdefault(cfg_name, {})[f"H{horizon}"] = {"error": "p3 failed"}
                continue

            # 5. 运行 p4 超参数搜索 (Optuna 模式)
            log(f"Step 4/4: 运行 p4-p6 (搜索+训练+评估)")
            r = run_cmd(
                [sys.executable, "-m", "experiments.prediction.p4_search_hparams",
                 "--horizon", str(horizon), "--mode", DEFAULT_MODE],
                f"p4_search_hparams [{cfg_name}] H={horizon} mode={DEFAULT_MODE}",
                timeout=3600,
            )
            if r.returncode != 0:
                log(f"X p4 失败，跳过 H={horizon}")
                results.setdefault(cfg_name, {})[f"H{horizon}"] = {"error": "p4 failed"}
                continue

            # 6. 运行 p5 训练
            log(f"Step 6/6: 运行 p5 训练")
            r = run_cmd(
                [sys.executable, "-m", "experiments.prediction.p5_train",
                 "--horizon", str(horizon), "--mode", DEFAULT_MODE,
                 "--seeds", str(DEFAULT_SEED)],
                f"p5_train [{cfg_name}] H={horizon} mode={DEFAULT_MODE} seed={DEFAULT_SEED}",
                timeout=3600,
            )
            if r.returncode != 0:
                log(f"X p5 失败，跳过 H={horizon}")
                results.setdefault(cfg_name, {})[f"H{horizon}"] = {"error": "p5 failed"}
                continue

            # p6 评估
            r = run_cmd(
                [sys.executable, "-m", "experiments.prediction.p6_evaluate",
                 "--horizon", str(horizon), "--mode", DEFAULT_MODE,
                 "--seeds", str(DEFAULT_SEED)],
                f"p6_evaluate [{cfg_name}] H={horizon} mode={DEFAULT_MODE}",
                timeout=600,
            )
            if r.returncode != 0:
                log(f"X p6 失败，跳过 H={horizon}")
                results.setdefault(cfg_name, {})[f"H{horizon}"] = {"error": "p6 failed"}
                continue

            # 收集结果
            metrics = collect_metrics(cfg_name, horizon, DEFAULT_MODE)
            results.setdefault(cfg_name, {})[f"H{horizon}"] = metrics
            log(f"  H={horizon} 结果: RMSE={metrics.get('RMSE')}, MAE={metrics.get('MAE')}, R2={metrics.get('R2')}")

        elapsed_cfg = time.time() - elapsed_cfg
        log(f"配置 {cfg_name} 完成，耗时 {elapsed_cfg:.0f}s")

    # 重置 _CFG_SUFFIX 为空
    reset_config_suffix()

    # ============================================================
    # 打印最终对比表格
    # ============================================================
    total_elapsed = time.time() - t_start

    print("\n" + "=" * 70)
    print("2配置预测结果对比表")
    print("=" * 70)

    # 打印每个 horizon 的结果
    for horizon in DEFAULT_HORIZONS:
        print(f"\n--- H={horizon} (提前 {horizon*15} 分钟) ---")
        rows = []
        for cfg in CONFIGS:
            name = cfg["name"]
            r = results.get(name, {}).get(f"H{horizon}", {})
            if "error" in r:
                rows.append({
                    "配置": name,
                    "方法": cfg["method"],
                    "σ": cfg["sigma"],
                    "RMSE": "ERROR",
                    "MAE": "ERROR",
                    "MAPE(%)": "ERROR",
                    "R2": "ERROR",
                    "持续性RMSE": "ERROR",
                    "相对提升(%)": "ERROR",
                })
            else:
                rows.append({
                    "配置": name,
                    "方法": cfg["method"],
                    "σ": cfg["sigma"],
                    "RMSE": f"{r.get('RMSE', 'N/A'):.4f}" if r.get('RMSE') is not None else "N/A",
                    "MAE": f"{r.get('MAE', 'N/A'):.4f}" if r.get('MAE') is not None else "N/A",
                    "MAPE(%)": f"{r.get('MAPE', 'N/A'):.2f}" if r.get('MAPE') is not None else "N/A",
                    "R2": f"{r.get('R2', 'N/A'):.4f}" if r.get('R2') is not None else "N/A",
                    "持续性RMSE": f"{r.get('persist_RMSE', 'N/A'):.4f}" if r.get('persist_RMSE') is not None else "N/A",
                    "相对提升(%)": f"{r.get('skill', 'N/A'):.2f}%" if r.get('skill') is not None else "N/A",
                })

        df_results = pd.DataFrame(rows)
        print(df_results.to_string(index=False))

    # 保存结果
    from experiments.common import PRED_RESULT_DIR
    PRED_RESULT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = PRED_RESULT_DIR / "4configs_comparison.csv"
    out_json = PRED_RESULT_DIR / "4configs_comparison.json"

    out_json.write_text(
        json.dumps({"configs": CONFIGS, "horizons": DEFAULT_HORIZONS, "results": results, "total_time_s": total_elapsed},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\n结果已保存: {out_json}")
    print(f"\n总耗时: {total_elapsed:.0f}s ({total_elapsed/60:.1f}min)")


if __name__ == "__main__":
    main()
