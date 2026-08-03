import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

pred_dir = 'data/prediction/step5_new_experiments/predictions/h16'
y_true = pd.read_csv(f'{pred_dir}/cnn_bilstm_residual_test.csv')['y_true'].values

models = [
    ('residual_optuna', 'cnn_bilstm_residual_optuna_test.csv'),
    ('combined_v2_improved', 'cnn_bilstm_combined_v2_improved_test.csv'),
]

print('=' * 70)
print('详细误差分析')
print('=' * 70)

for name, file in models:
    df = pd.read_csv(f'{pred_dir}/{file}')
    y_pred = df['y_pred'].values
    errors = y_true - y_pred

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # 误差分布
    abs_errors = np.abs(errors)
    large_errors = np.sum(abs_errors > 0.1)
    small_errors = np.sum(abs_errors <= 0.05)

    # 方差分析
    ss_res = np.sum(errors**2)
    ss_tot = np.sum((y_true - y_true.mean())**2)

    print(f'\n{name}:')
    print(f'  RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}')
    print(f'  大误差(>0.1)数量: {large_errors} ({100*large_errors/len(errors):.1f}%)')
    print(f'  小误差(<=0.05)数量: {small_errors} ({100*small_errors/len(errors):.1f}%)')
    print(f'  SS_res={ss_res:.4f}, SS_tot={ss_tot:.4f}')
    print(f'  真实值均值={y_true.mean():.4f}, 预测值均值={y_pred.mean():.4f}')

print('\n' + '=' * 70)
print('原因分析:')
print('=' * 70)
print('1. MAE下降 = 平均绝对误差减少，说明预测更"接近"真实值')
print('2. R2下降 = SS_res/SS_tot比值增大')
print('3. 可能原因: CombinedV2减少了部分大误差，但引入更多小误差')
print('4. Huber损失对异常值鲁棒，倾向于"平均"预测而非过拟合')
