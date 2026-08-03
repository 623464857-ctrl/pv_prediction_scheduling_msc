import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json
import os

# 基准数据目录
base_dir = 'data/prediction/step5_new_experiments/predictions'

horizons = ['h1', 'h4', 'h16']

for horizon in horizons:
    print(f'\n{"="*80}')
    print(f'{horizon.upper()} 预测结果对比')
    print(f'{"="*80}')

    pred_dir = f'{base_dir}/{horizon}'

    # 真实值
    y_true = pd.read_csv(f'{pred_dir}/cnn_bilstm_residual_test.csv')['y_true'].values

    # 分段掩码
    peak_mask = y_true > 0.3
    low_mask = y_true < 0.05

    models = [
        ('residual_optuna', 'cnn_bilstm_residual_optuna_test.csv'),
        ('mse_improved', 'cnn_bilstm_mse_improved_test.csv'),
        ('asymmetric_mse_improved', 'cnn_bilstm_asymmetric_mse_improved_test.csv'),
    ]

    # 检查combined_v2是否存在
    if os.path.exists(f'{pred_dir}/cnn_bilstm_combined_v2_improved_test.csv'):
        models.append(('combined_v2_improved', 'cnn_bilstm_combined_v2_improved_test.csv'))

    print(f"\n{'模型':<25} {'RMSE':>10} {'MAE':>10} {'R2':>10} {'Peak RMSE':>12} {'Low RMSE':>12}")
    print('-' * 90)

    for name, file in models:
        df = pd.read_csv(f'{pred_dir}/{file}')
        y_pred = df['y_pred'].values

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        peak_rmse = np.sqrt(mean_squared_error(y_true[peak_mask], y_pred[peak_mask]))
        low_rmse = np.sqrt(mean_squared_error(y_true[low_mask], y_pred[low_mask]))

        print(f"{name:<25} {rmse:>10.4f} {mae:>10.4f} {r2:>10.4f} {peak_rmse:>12.4f} {low_rmse:>12.4f}")

print('\n' + '='*80)
