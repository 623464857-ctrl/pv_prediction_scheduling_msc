"""生成预测曲线对比图 - H1/H4/H16"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 数据路径
PRED_DIR = Path('data/prediction/step5_new_experiments/predictions')
FIG_DIR = Path('data/prediction/step5_new_experiments/figures')

# 颜色映射 - 使用不同颜色区分不同方法
COLORS = {
    'cnn_bilstm_physics': '#E74C3C',      # 红色
    'cnn_bilstm_residual': '#3498DB',     # 蓝色
    'lightgbm': '#2ECC71',                # 绿色
    'ridge': '#9B59B6',                   # 紫色
    'persistence': '#F39C12',             # 橙色
}

LINE_STYLES = {
    'y_true': '-',
    'y_pred': '--',
}

# 读取数据
def load_predictions(h, model='cnn_bilstm_physics'):
    """加载指定horizon和模型的预测数据"""
    path = PRED_DIR / f'h{h}' / f'{model}_test.csv'
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=['timestamp'])
    return df

# 创建图形 - 1行3列
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('PV Power Prediction: Ground Truth vs Predictions', fontsize=14, fontweight='bold')

horizons = [1, 4, 16]
model_name = 'cnn_bilstm_physics'

for idx, h in enumerate(horizons):
    ax = axes[idx]
    df = load_predictions(h, model_name)
    
    if df is None:
        ax.set_title(f'Horizon {h} (No Data)')
        continue
    
    # 选择一天的数据进行可视化 (约96个点=24小时*4)
    n_points = min(288, len(df))  # 显示3天数据
    x = np.arange(n_points)
    
    y_true = df['y_true'].values[:n_points]
    y_pred = df['y_pred'].values[:n_points]
    
    # 绘制真实值 - 黑色实线
    ax.plot(x, y_true, 'k-', linewidth=1.5, label='Ground Truth', alpha=0.9)
    
    # 绘制预测值 - 带颜色虚线
    ax.plot(x, y_pred, color=COLORS[model_name], linestyle='--', 
            linewidth=1.2, label=f'{model_name}', alpha=0.85)
    
    # 设置标题和标签
    ax.set_title(f'Horizon {h}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Time Step (15 min intervals)', fontsize=10)
    ax.set_ylabel('Normalized Power', fontsize=10)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, n_points)
    
    # 添加统计信息
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    ax.text(0.02, 0.98, f'RMSE: {rmse:.4f}\nMAE: {mae:.4f}', 
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
output_path = FIG_DIR / 'prediction_curves_comparison.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f'Saved: {output_path}')

# ============================================
# 额外：多方法对比图
# ============================================
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
fig2.suptitle('Multi-Model Comparison (Horizon H1/H4/H16)', fontsize=14, fontweight='bold')

models_to_compare = ['cnn_bilstm_physics', 'cnn_bilstm_residual', 'lightgbm', 'ridge']
model_labels = ['CNN-BiLSTM+Physics', 'CNN-BiLSTM+Residual', 'LightGBM', 'Ridge']
colors_multi = ['#E74C3C', '#3498DB', '#2ECC71', '#9B59B6']

for idx, h in enumerate(horizons):
    ax = axes2[idx]
    df_true = load_predictions(h, 'cnn_bilstm_physics')
    
    if df_true is None:
        continue
    
    # 真实值 - 黑色实线
    n_points = min(288, len(df_true))
    x = np.arange(n_points)
    y_true = df_true['y_true'].values[:n_points]
    ax.plot(x, y_true, 'k-', linewidth=2, label='Ground Truth', alpha=0.9)
    
    # 各模型预测值 - 彩色虚线
    for model, label, color in zip(models_to_compare, model_labels, colors_multi):
        df_pred = load_predictions(h, model)
        if df_pred is not None:
            y_pred = df_pred['y_pred'].values[:n_points]
            ax.plot(x, y_pred, color=color, linestyle='--', 
                   linewidth=1, label=label, alpha=0.7)
    
    ax.set_title(f'Horizon {h}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Time Step', fontsize=10)
    ax.set_ylabel('Normalized Power', fontsize=10)
    ax.legend(loc='upper right', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, n_points)

plt.tight_layout()
output_path2 = FIG_DIR / 'multi_model_comparison.png'
plt.savefig(output_path2, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f'Saved: {output_path2}')

print('\nDone!')
