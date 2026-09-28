"""
亳州光伏预测 - 从模型加载并绘图

加载已训练好的模型，生成预测并绘制结果图。
使用方法:
    python plot_bozhou_from_model.py
    
可选参数:
    --models-dir   指定模型目录 (默认: data/prediction/step2_hyperparameter_search/models)
    --samples-dir  指定样本数据目录 (默认: data/prediction/step2_hyperparameter_search/samples)
    --output       指定输出图片路径
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 路径配置
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODELS_DIR = PROJECT_ROOT / "data" / "prediction" / "step2_hyperparameter_search" / "models"
DEFAULT_SAMPLES_DIR = PROJECT_ROOT / "data" / "prediction" / "step2_hyperparameter_search" / "samples"
OUTPUT_DIR = PROJECT_ROOT / "result" / "figures"


def load_y_scaler(sample_dir: Path) -> StandardScaler:
    """从样本目录加载 y scaler 参数"""
    scaler_file = sample_dir / "scaler_params.json"
    
    if not scaler_file.exists():
        raise FileNotFoundError(f"Scaler参数文件不存在: {scaler_file}")
    
    with open(scaler_file, encoding='utf-8') as f:
        params = json.load(f)
    
    scaler = StandardScaler()
    scaler.mean_ = np.array(params["y_mean"])
    scaler.scale_ = np.array(params["y_scale"])
    scaler.n_features_in_ = len(params["y_mean"])
    scaler.n_samples_seen_ = None
    
    return scaler


def load_sample_data(sample_dir: Path) -> dict:
    """加载样本数据"""
    required_files = [
        "X_test_seq.npy", "y_test.npy", 
        "X_train_seq.npy", "y_train.npy",
        "X_val_seq.npy", "y_val.npy",
        "meta.json", "test_timestamps.csv"
    ]
    
    data = {}
    for fname in required_files:
        fpath = sample_dir / fname
        if not fpath.exists():
            raise FileNotFoundError(f"样本文件不存在: {fpath}")
        
        if fname.endswith('.npy'):
            data[fname.replace('.npy', '')] = np.load(fpath)
        elif fname.endswith('.json'):
            with open(fpath, encoding='utf-8') as f:
                data[fname.replace('.json', '')] = json.load(f)
        elif fname.endswith('.csv'):
            data[fname.replace('.csv', '')] = pd.read_csv(fpath)
    
    return data


def load_model(horizon: int, models_dir: Path, device: str = None) -> torch.nn.Module:
    """加载训练好的模型"""
    model_file = models_dir / f"bozhou_h{horizon}" / "bozhou_cnn_bilstm_seed42.pt"
    
    if not model_file.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_file}")
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 导入模型定义
    sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "prediction" / "step3_deep_learning"))
    from exp_p04_models import build_model
    
    # 从 meta.json 获取模型参数
    sample_dir = DEFAULT_SAMPLES_DIR / f"bozhou_h{horizon}_lb{16 if horizon == 1 else (48 if horizon == 4 else 96)}"
    with open(sample_dir / "meta.json", encoding='utf-8') as f:
        meta = json.load(f)
    
    # 加载最佳参数
    metrics_dir = PROJECT_ROOT / "data" / "prediction" / "step2_hyperparameter_search" / "metrics" / f"bozhou_h{horizon}"
    optuna_file = metrics_dir / "bozhou_cnn_bilstm_optuna.json"
    
    if optuna_file.exists():
        with open(optuna_file, encoding='utf-8') as f:
            optuna_data = json.load(f)
        model_params = optuna_data.get("best_params", {})
    else:
        model_params = {
            "hidden_size": 64, "num_layers": 2, "dropout": 0.2,
            "cnn_channels": [32, 64], "kernel_size": 3,
        }
    
    model = build_model(
        "cnn_bilstm",
        n_features=meta["n_features"],
        seq_len=meta["lookback"],
        horizon=horizon,
        **model_params
    ).to(device)
    
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.eval()
    
    return model


def predict_with_model(model, X_test: np.ndarray, device: str) -> np.ndarray:
    """使用模型进行预测"""
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_test).to(device)
        predictions = model(X_tensor).cpu().numpy()
    return predictions


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """计算评估指标"""
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()
    
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    
    return {"RMSE": rmse, "MAE": mae, "R2": r2, "MAPE": mape}


def plot_model_predictions(results: dict, output_path: Path = None):
    """绘制模型预测结果"""
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    fig, axes = plt.subplots(3, 1, figsize=(18, 12), sharex=True)
    plt.subplots_adjust(hspace=0.15)
    
    horizon_labels = {
        1: "H=1 (15分钟预测)", 
        4: "H=4 (1小时预测)", 
        16: "H=16 (4小时预测)"
    }
    colors = {1: "#2E86AB", 4: "#A23B72", 16: "#F18F01"}
    
    for idx, horizon in enumerate(sorted(results.keys())):
        ax = axes[idx]
        res = results[horizon]
        
        y_true = res["y_true"]
        y_pred = res["y_pred"]
        timestamps = res["timestamps"]
        metrics = res["metrics"]
        
        ax.plot(timestamps, y_true, 
                label="真实值 (Actual)", 
                color="black", linewidth=1.5, linestyle='-', alpha=0.95)
        ax.plot(timestamps, y_pred, 
                label="预测值 (Predicted)", 
                color=colors.get(horizon, "blue"), 
                linewidth=1.2, linestyle='--', alpha=0.85)
        
        ax.set_ylabel("功率 (kW)", fontsize=12, fontweight='bold')
        ax.set_title(f"亳州光伏预测 - {horizon_labels.get(horizon, f'H={horizon}')}  |  "
                    f"RMSE={metrics['RMSE']:.4f}, MAE={metrics['MAE']:.4f}, R²={metrics['R2']:.4f}",
                    fontsize=13, fontweight="bold", pad=10)
        
        ax.legend(loc="upper right", fontsize=11, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.tick_params(axis='x', rotation=30, labelsize=10)
        ax.tick_params(axis='y', labelsize=10)
        
        y_min = min(y_true.min(), y_pred.min())
        y_max = max(y_true.max(), y_pred.max())
        y_range = y_max - y_min
        if y_range > 0:
            ax.set_ylim(y_min - 0.05 * y_range, y_max + 0.05 * y_range)
    
    axes[-1].set_xlabel("时间 (Time)", fontsize=12, fontweight='bold')
    
    fig.suptitle("亳州光伏电站功率预测 - CNN-BiLSTM 模型\nBozhou PV Station Power Prediction - CNN-BiLSTM Model", 
                 fontsize=15, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    if output_path is None:
        output_path = OUTPUT_DIR / "bozhou_model_predictions.png"
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"图片已保存: {output_path}")
    
    pdf_path = output_path.with_suffix('.pdf')
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f"PDF已保存: {pdf_path}")
    
    plt.close()
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="亳州光伏预测 - 从模型绘图")
    parser.add_argument("--models-dir", type=str, default=None,
                        help="模型目录路径")
    parser.add_argument("--samples-dir", type=str, default=None,
                        help="样本数据目录路径")
    parser.add_argument("--output", type=str, default=None,
                        help="输出图片路径")
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 4, 16],
                        help="预测步长列表 (默认: 1 4 16)")
    
    args = parser.parse_args()
    
    # 设置路径
    models_dir = Path(args.models_dir) if args.models_dir else DEFAULT_MODELS_DIR
    samples_dir = Path(args.samples_dir) if args.samples_dir else DEFAULT_SAMPLES_DIR
    
    print("=" * 60)
    print("亳州光伏预测 - 从模型加载并绘图")
    print(f"模型目录: {models_dir}")
    print(f"样本目录: {samples_dir}")
    print(f"预测步长: {args.horizons}")
    print("=" * 60)
    
    # 检测设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    results = {}
    
    for horizon in args.horizons:
        print(f"\n处理 H={horizon}...")
        
        try:
            # 确定 lookback
            lookback_map = {1: 16, 4: 48, 16: 96}
            lookback = lookback_map.get(horizon, 16)
            sample_dir = samples_dir / f"bozhou_h{horizon}_lb{lookback}"
            
            # 加载数据和模型
            data = load_sample_data(sample_dir)
            model = load_model(horizon, models_dir, device)
            y_scaler = load_y_scaler(sample_dir)
            
            # 生成预测
            y_pred_scaled = predict_with_model(model, data["X_test_seq"], device)
            y_pred = y_scaler.inverse_transform(y_pred_scaled)
            y_true = y_scaler.inverse_transform(data["y_test"])
            
            # 计算指标
            metrics = compute_metrics(y_true, y_pred)
            print(f"  H={horizon}: RMSE={metrics['RMSE']:.4f}, MAE={metrics['MAE']:.4f}, R²={metrics['R2']:.4f}")
            
            # 处理时间戳
            timestamps = pd.to_datetime(data["test_timestamps"]["timestamp"])
            
            results[horizon] = {
                "y_true": y_true.ravel(),
                "y_pred": y_pred.ravel(),
                "timestamps": timestamps,
                "metrics": metrics
            }
            
        except FileNotFoundError as e:
            print(f"  警告: {e}")
            continue
        except Exception as e:
            print(f"  错误: {e}")
            continue
    
    if results:
        output_path = Path(args.output) if args.output else None
        plot_model_predictions(results, output_path)
        print("\n绘图完成!")
        
        # 打印汇总
        print("\n" + "=" * 60)
        print("预测结果汇总:")
        print("=" * 60)
        for horizon, res in sorted(results.items()):
            m = res["metrics"]
            print(f"  H={horizon:2d}: RMSE={m['RMSE']:.4f}, MAE={m['MAE']:.4f}, R²={m['R2']:.4f}")
        print("=" * 60)
    else:
        print("\n没有成功处理任何 horizon，请检查文件路径。")


if __name__ == "__main__":
    main()
