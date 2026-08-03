import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

pred_dir = 'data/prediction/step5_new_experiments/predictions/h16'

# 读取真实值
y_true = pd.read_csv(f'{pred_dir}/cnn_bilstm_residual_test.csv')['y_true'].values

models = [
    ('residual_optuna', 'cnn_bilstm_residual_optuna_test.csv'),
    ('mse_improved', 'cnn_bilstm_mse_improved_test.csv'),
    ('asymmetric_mse_improved', 'cnn_bilstm_asymmetric_mse_improved_test.csv'),
    ('combined_v2_improved', 'cnn_bilstm_combined_v2_improved_test.csv'),
]

print('=' * 70)
print('H16 整体指标对比')
print('=' * 70)
print(f"{'模型':<25} {'RMSE':>10} {'MAE':>10} {'R2':>10}")
print('-' * 70)

for name, file in models:
    df = pd.read_csv(f'{pred_dir}/{file}')
    y_pred = df['y_pred'].values
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{name:<25} {rmse:>10.4f} {mae:>10.4f} {r2:>10.4f}")

print('=' * 70)
