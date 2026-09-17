"""
快速对比实验：StandardScaler vs Yeo-Johnson + StandardScaler
==============================================================
使用项目当前模型: CNN-BiLSTM (深度学习)

实验目的：验证对于 CNN-BiLSTM 模型，仅使用 StandardScaler 是否效果
         与 Yeo-Johnson + StandardScaler 等价

使用方法：
  python -m experiments.prediction.step1_preprocessing.exp_scaler_comparison

输出：
  - 各预处理方法的 CV RMSE/MAE/R²
  - 两种方法的对比表格
  - 差异统计（是否在统计误差范围内）
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import warnings
import json
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import yeojohnson, yeojohnson_normmax

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

# ============================================================
# 0. 中文字体配置
# ============================================================
import matplotlib.font_manager as fm
CHINESE_FONTS = ["Microsoft YaHei", "SimHei", "KaiTi", "STKaiti", "Noto Sans SC"]
_font_found = None
for font_name in CHINESE_FONTS:
    available = [f.name for f in fm.fontManager.ttflist]
    if font_name in available:
        _font_found = font_name
        break
if _font_found is None:
    _font_found = fm.findfont(fm.FontProperties()).split("\\")[-1].replace(".ttf", "")
plt.rcParams["font.family"] = _font_found
plt.rcParams["axes.unicode_minus"] = False
print(f"[Font] Using: {_font_found}")

matplotlib.use("Agg")

# ============================================================
# 1. 配置
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = PROJECT_ROOT / "data/prediction/step1_preprocessing/processed/mingyuehu_long.csv"
OUTPUT_DIR = PROJECT_ROOT / "data/prediction/step1_preprocessing/charts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 目标变量和特征
TARGET = "power_pu"
EXCLUDE_COLS = [
    "timestamp", "power_kw", "power_pu",
    "is_daytime", "is_potential_shutdown",
    "power_ramp_15m_kw", "power_ramp_15m_pu",
    "power_kw_negative_clipped_flag",
    "imputed_feature_count", "raw_issue_count", "data_quality_score"
]

# 时间序列 CV 配置
N_FOLDS = 5
TEST_SIZE = 192  # 约 2 天数据

# CNN-BiLSTM 模型参数
MODEL_PARAMS = {
    "hidden_size": 64,
    "num_layers": 2,
    "dropout": 0.2,
    "cnn_channels": (32, 64),
    "kernel_size": 3,
}
TRAIN_PARAMS = {
    "batch_size": 64,
    "lr": 0.001,
    "max_epochs": 30,
    "patience": 5,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

print("=" * 60)
print("Scalers Comparison: StandardScaler vs Yeo-Johnson + StandardScaler")
print(f"Model: CNN-BiLSTM (hidden={MODEL_PARAMS['hidden_size']}, layers={MODEL_PARAMS['num_layers']})")
print(f"Device: {TRAIN_PARAMS['device']}")
print("=" * 60)

# ============================================================
# 2. CNN-BiLSTM 模型定义 (从 exp_p04_models.py 复制)
# ============================================================
class CNN_BiLSTM(nn.Module):
    def __init__(
        self,
        n_features: int,
        seq_len: int,
        horizon: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        cnn_channels: list[int] | tuple[int, ...] = (32, 64),
        kernel_size: int = 3,
    ):
        super().__init__()
        self.n_features = n_features
        self.seq_len = seq_len
        self.horizon = horizon

        cnn_layers = []
        in_ch = n_features
        for out_ch in cnn_channels:
            cnn_layers.append(
                nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size, padding=kernel_size // 2)
            )
            cnn_layers.append(nn.BatchNorm1d(out_ch))
            cnn_layers.append(nn.ReLU())
            cnn_layers.append(nn.Dropout(dropout))
            in_ch = out_ch
        self.cnn = nn.Sequential(*cnn_layers)

        self.lstm = nn.LSTM(
            input_size=cnn_channels[-1] if cnn_channels else n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        lstm_output_size = hidden_size * 2
        self.fc = nn.Linear(lstm_output_size, horizon)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out)
        return out


# ============================================================
# 3. 加载数据
# ============================================================
print(f"\n[1] Loading data: {DATA_PATH}")
df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

# 只使用白天数据
df = df[df["is_daytime"] == 1].copy()
print(f"    Daytime samples: {len(df)}")

# 特征列
feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
num_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
feature_cols = num_cols
print(f"    Features: {len(feature_cols)}")

X = df[feature_cols].copy()
y = df[TARGET].copy()

# ============================================================
# 4. 预处理方法
# ============================================================
def preprocess_std_only(X_train, X_test, y_train, y_test):
    """Method A: StandardScaler only"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel()
    
    return X_train_scaled, X_test_scaled, y_train_scaled, y_scaler, None


def preprocess_yj_std(X_train, X_test, y_train, y_test):
    """Method B: Yeo-Johnson + StandardScaler"""
    yj_lambdas = {}
    X_train_yj = pd.DataFrame(index=X_train.index)
    X_test_yj = pd.DataFrame(index=X_test.index)
    
    for col in X_train.columns:
        x_train = X_train[col].values.astype(float)
        x_test = X_test[col].values.astype(float)
        lm = yeojohnson_normmax(x_train)
        yj_lambdas[col] = lm
        X_train_yj[col] = yeojohnson(x_train, lm)
        X_test_yj[col] = yeojohnson(x_test, lm)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_yj)
    X_test_scaled = scaler.transform(X_test_yj)
    
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel()
    
    return X_train_scaled, X_test_scaled, y_train_scaled, y_scaler, yj_lambdas


# ============================================================
# 5. 创建序列样本
# ============================================================
def create_sequences(X, y, seq_len):
    """创建时间序列样本"""
    X_seq, y_seq = [], []
    y_values = y.values if hasattr(y, 'values') else y
    for i in range(seq_len, len(X)):
        X_seq.append(X[i-seq_len:i])
        y_seq.append(y_values[i])
    return np.array(X_seq), np.array(y_seq)


def create_cv_samples(X, y, train_end, test_end, seq_len, scaler_params):
    """创建 CV 样本"""
    X_train = X[:train_end]
    X_test = X[train_end:test_end]
    y_train = y[:train_end]
    y_test = y[train_end:test_end]
    
    # 预处理
    X_train_s, X_test_s, y_train_s, y_scaler, yj_lambdas = scaler_params(X_train, X_test, y_train, y_test)
    
    # 创建序列
    X_train_seq, y_train_seq = create_sequences(X_train_s, pd.Series(y_train_s), seq_len)
    X_test_seq, y_test_seq = create_sequences(X_test_s, pd.Series(y_test_seq), seq_len) if len(X_test_s) > seq_len else (np.array([]), np.array([]))
    
    return X_train_seq, y_train_seq, X_test_seq, y_test_seq, y_scaler


# ============================================================
# 6. 训练 CNN-BiLSTM
# ============================================================
def train_cnn_bilstm(X_train, y_train, X_val, y_val, seq_len, n_features, horizon, device):
    """训练 CNN-BiLSTM 模型"""
    model = CNN_BiLSTM(
        n_features=n_features,
        seq_len=seq_len,
        horizon=horizon,
        **MODEL_PARAMS
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=TRAIN_PARAMS["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
    
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train).unsqueeze(-1) if y_train.ndim == 1 else torch.FloatTensor(y_train)
    )
    train_loader = DataLoader(train_dataset, batch_size=TRAIN_PARAMS["batch_size"], shuffle=True)
    
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    
    for epoch in range(TRAIN_PARAMS["max_epochs"]):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs[:, -1, 0] if outputs.shape[1] > 1 else outputs[:, 0], 
                           y_batch.squeeze())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # 验证
        model.eval()
        with torch.no_grad():
            if len(X_val) > 0:
                X_val_t = torch.FloatTensor(X_val).to(device)
                y_val_t = torch.FloatTensor(y_val).to(device)
                val_pred = model(X_val_t)
                val_loss = criterion(val_pred[:, -1, 0] if val_pred.shape[1] > 1 else val_pred[:, 0],
                                   y_val_t.squeeze()).item()
            else:
                val_loss = train_loss / len(train_loader)
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= TRAIN_PARAMS["patience"]:
                break
    
    model.load_state_dict(best_state)
    return model


def predict_cnn_bilstm(model, X_test, device):
    """使用 CNN-BiLSTM 预测"""
    model.eval()
    with torch.no_grad():
        X_t = torch.FloatTensor(X_test).to(device)
        pred = model(X_t)
        if pred.shape[1] > 1:
            pred = pred[:, -1, 0]
        else:
            pred = pred[:, 0]
    return pred.cpu().numpy()


# ============================================================
# 7. 时间序列 CV
# ============================================================
SEQ_LEN = 16  # lookback = 16
HORIZON = 1   # 预测步数 = 1

def time_series_cv_dl(X, y, preprocess_fn, n_folds=N_FOLDS, test_size=TEST_SIZE):
    """时间序列滚动交叉验证 (用于深度学习)"""
    n_samples = len(X)
    results = []
    device = torch.device(TRAIN_PARAMS["device"])
    n_features = X.shape[1]
    
    for fold in range(n_folds):
        test_start = n_samples - (fold + 1) * test_size
        test_end = n_samples - fold * test_size
        
        if test_start < int(n_samples * 0.5):
            continue
        
        train_end = test_start
        X_train_full = X[:train_end].values if hasattr(X, 'values') else X[:train_end]
        X_test = X[train_end:test_end].values if hasattr(X, 'values') else X[train_end:test_end]
        y_train_full = y.iloc[:train_end]
        y_test = y.iloc[train_end:test_end]
        
        # 预处理
        X_train_s, X_test_s, y_train_s, y_scaler, _ = preprocess_fn(
            pd.DataFrame(X_train_full, columns=X.columns),
            pd.DataFrame(X_test, columns=X.columns),
            y_train_full,
            y_test
        )
        
        # 创建序列样本
        X_train_seq, y_train_seq = create_sequences(X_train_s, pd.Series(y_train_s), SEQ_LEN)
        
        # 分割训练/验证集 (最后 20% 作为验证)
        val_size = int(len(X_train_seq) * 0.2)
        if val_size > 0:
            X_train = X_train_seq[:-val_size]
            y_train = y_train_seq[:-val_size]
            X_val = X_train_seq[-val_size:]
            y_val = y_train_seq[-val_size:]
        else:
            X_train = X_train_seq
            y_train = y_train_seq
            X_val = np.array([])
            y_val = np.array([])
        
        if len(X_train) < 50 or len(X_test_s) <= SEQ_LEN:
            print(f"    Fold {fold+1}: Skipped (insufficient data)")
            continue
        
        # 训练模型
        model = train_cnn_bilstm(
            X_train, y_train, X_val, y_val,
            SEQ_LEN, n_features, HORIZON, device
        )
        
        # 预测
        if len(X_test_s) > SEQ_LEN:
            X_test_seq, _ = create_sequences(X_test_s, y_test.values, SEQ_LEN)
            y_pred_scaled = predict_cnn_bilstm(model, X_test_seq, device)
            y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
            y_true = y_test.iloc[SEQ_LEN:].values
        else:
            y_pred = np.array([])
            y_true = np.array([])
        
        if len(y_pred) < 10:
            print(f"    Fold {fold+1}: Skipped (test samples < 10)")
            continue
        
        # 计算指标
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        results.append({
            "fold": fold + 1,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "n_train": len(X_train),
            "n_test": len(y_true)
        })
        print(f"    Fold {fold+1}: RMSE={rmse:.6f}, MAE={mae:.6f}, R2={r2:.6f}")
    
    return pd.DataFrame(results)


# ============================================================
# 8. 运行实验
# ============================================================
methods = {
    "StandardScaler only": preprocess_std_only,
    "Yeo-Johnson + StandardScaler": preprocess_yj_std,
}

print(f"\n[2] Running Time Series CV (n_folds={N_FOLDS}, test_size={TEST_SIZE})...")
print("-" * 60)

all_results = {}
for name, fn in methods.items():
    print(f"\n>>> Method: {name}")
    cv_results = time_series_cv_dl(X, y, fn)
    all_results[name] = cv_results
    
    if len(cv_results) > 0:
        mean_rmse = cv_results["rmse"].mean()
        std_rmse = cv_results["rmse"].std()
        mean_mae = cv_results["mae"].mean()
        std_mae = cv_results["mae"].std()
        mean_r2 = cv_results["r2"].mean()
        print(f"    Summary: RMSE={mean_rmse:.6f}+/-{std_rmse:.6f}, MAE={mean_mae:.6f}+/-{std_mae:.6f}, R2={mean_r2:.6f}")

# ============================================================
# 9. 对比汇总
# ============================================================
print("\n" + "=" * 60)
print("[3] Comparison Summary")
print("=" * 60)

summary = []
for name, cv in all_results.items():
    if len(cv) > 0:
        summary.append({
            "Method": name,
            "RMSE (mean+/-std)": f"{cv['rmse'].mean():.6f} +/- {cv['rmse'].std():.6f}",
            "MAE (mean+/-std)": f"{cv['mae'].mean():.6f} +/- {cv['mae'].std():.6f}",
            "R2 (mean)": f"{cv['r2'].mean():.6f}",
            "RMSE Value": cv['rmse'].mean(),
            "MAE Value": cv['mae'].mean(),
            "R2 Value": cv['r2'].mean(),
        })

summary_df = pd.DataFrame(summary)

if len(summary_df) == 0:
    print("No results to compare!")
    exit()

# 找最佳方法
best_idx = summary_df["RMSE Value"].idxmin()
best_method = summary_df.loc[best_idx, "Method"]
baseline_rmse = summary_df[summary_df["Method"] == "StandardScaler only"]["RMSE Value"].values[0] if "StandardScaler only" in summary_df["Method"].values else summary_df["RMSE Value"].min()

print(f"\nBaseline: StandardScaler only")
print(f"\n{'Method':<35} {'RMSE':>20} {'MAE':>20} {'R2':>12} {'vs Baseline':>15}")
print("-" * 105)

for _, row in summary_df.iterrows():
    diff_rmse = ((row["RMSE Value"] - baseline_rmse) / baseline_rmse) * 100
    diff_str = f"{diff_rmse:+.2f}%" if diff_rmse != 0 else "baseline"
    marker = " *" if row["Method"] == best_method else ""
    print(f"{row['Method']:<35} {row['RMSE (mean+/-std)']:>20} {row['MAE (mean+/-std)']:>20} {row['R2 (mean)']:>12} {diff_str:>15}{marker}")

# ============================================================
# 10. 统计显著性检验
# ============================================================
print("\n[4] Statistical Significance Test (Paired t-test)")
print("-" * 60)

from scipy.stats import ttest_rel

if len(all_results) >= 2:
    method_names = list(all_results.keys())
    if len(all_results[method_names[0]]) > 0 and len(all_results[method_names[1]]) > 0:
        cv1 = all_results[method_names[0]]
        cv2 = all_results[method_names[1]]
        
        min_len = min(len(cv1), len(cv2))
        if min_len >= 2:
            t_stat, p_value = ttest_rel(
                cv1["rmse"].values[:min_len],
                cv2["rmse"].values[:min_len]
            )
            diff_mean = cv2["rmse"].mean() - cv1["rmse"].mean()
            print(f"{method_names[0]} vs {method_names[1]}:")
            print(f"  RMSE diff mean: {diff_mean:.6f}")
            print(f"  p-value: {p_value:.4f} {'(Significant, p<0.05)' if p_value < 0.05 else '(Not significant, p>=0.05)'}")

# ============================================================
# 11. 可视化
# ============================================================
print("\n[5] Generating visualizations...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(
    "StandardScaler vs Yeo-Johnson + StandardScaler\n"
    f"(CNN-BiLSTM, Time Series {N_FOLDS}-Fold Rolling CV)",
    fontsize=12, fontweight="bold"
)

colors = ["#2196F3", "#FF5722"]
method_names = list(all_results.keys())

# 11a: 各折 RMSE 对比
ax = axes[0]
if len(all_results[method_names[0]]) > 0:
    fold_nums = all_results[method_names[0]]["fold"].values
    x = np.arange(len(fold_nums))
    width = 0.35
    for i, (name, cv) in enumerate(all_results.items()):
        if len(cv) > 0:
            ax.bar(x + i * width, cv["rmse"].values, width, label=name, color=colors[i], alpha=0.8)
    ax.set_xlabel("Fold")
    ax.set_ylabel("RMSE")
    ax.set_title("(a) RMSE per Fold")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([f"Fold {f}" for f in fold_nums])
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

# 11b: RMSE 箱线图
ax = axes[1]
rmse_data = []
labels = []
for name in method_names:
    cv = all_results[name]
    if len(cv) > 0:
        rmse_data.append(cv["rmse"].values)
        labels.append(name.split()[0])  # 简写
if len(rmse_data) > 0:
    bp = ax.boxplot(rmse_data, labels=labels, patch_artist=True)
    for patch, color in zip(bp["boxes"], colors[:len(rmse_data)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    ax.set_ylabel("RMSE")
    ax.set_title("(b) RMSE Distribution")
    ax.grid(axis="y", alpha=0.3)

# 11c: 差异图
ax = axes[2]
if len(all_results[method_names[0]]) > 0 and len(all_results[method_names[1]]) > 0:
    cv1 = all_results[method_names[0]]
    cv2 = all_results[method_names[1]]
    min_len = min(len(cv1), len(cv2))
    if min_len >= 2:
        diff = cv2["rmse"].values[:min_len] - cv1["rmse"].values[:min_len]
        ax.bar(range(1, min_len + 1), diff, color="#FF5722", alpha=0.8)
        ax.axhline(0, color="gray", lw=1.5, ls="--")
        ax.set_xlabel("Fold")
        ax.set_ylabel("RMSE diff (YJ+Std - Std)")
        ax.set_title("(c) Difference vs Baseline\n(positive=StandardScaler better)")
        ax.grid(axis="y", alpha=0.3)

fig.tight_layout()
fig.savefig(OUTPUT_DIR / "fig_scaler_comparison_cnn_results.png", dpi=150, bbox_inches="tight")
print(f"[Saved] fig_scaler_comparison_cnn_results.png")

# ============================================================
# 12. 最终结论
# ============================================================
print("\n" + "=" * 60)
print("[6] Final Conclusion")
print("=" * 60)

if "StandardScaler only" in summary_df["Method"].values and "Yeo-Johnson + StandardScaler" in summary_df["Method"].values:
    std_rmse = summary_df[summary_df["Method"] == "StandardScaler only"]["RMSE Value"].values[0]
    yj_rmse = summary_df[summary_df["Method"] == "Yeo-Johnson + StandardScaler"]["RMSE Value"].values[0]
    yj_improvement = ((std_rmse - yj_rmse) / std_rmse) * 100
    
    if abs(yj_improvement) < 1.0:
        conclusion = f"""
[Conclusion] RMSE difference between methods is {abs(yj_improvement):.2f}% (< 1%), within statistical error.
For CNN-BiLSTM with BatchNorm, skewness correction (Yeo-Johnson) provides NO significant improvement.

-> Recommended: StandardScaler only
-> Reason: Simpler, no inverse transform needed, equivalent to other methods
"""
    elif yj_improvement > 0:
        conclusion = f"""
[Conclusion] Yeo-Johnson + StandardScaler shows {yj_improvement:.2f}% improvement over StandardScaler.
The difference is small but Yeo-Johnson provides additional skewness correction value.

-> Suggestion: Consider using StandardScaler only for simplicity
"""
    else:
        conclusion = f"""
[Conclusion] StandardScaler performs best among the compared methods.
Yeo-Johnson + StandardScaler shows {abs(yj_improvement):.2f}% worse performance.

-> Recommended: StandardScaler only
"""
else:
    conclusion = "[Conclusion] Insufficient data for comparison."

print(conclusion)
print("=" * 60)

# 保存结果
summary_df.to_csv(OUTPUT_DIR / "scaler_comparison_cnn_summary.csv", index=False)
print(f"\n[Saved] Results: scaler_comparison_cnn_summary.csv")
