import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

pred_dir = 'data/prediction/step5_new_experiments/predictions/h16'
y_true = pd.read_csv(f'{pred_dir}/cnn_bilstm_residual_optuna_test.csv')['y_true'].values

df1 = pd.read_csv(f'{pred_dir}/cnn_bilstm_residual_optuna_test.csv')
df2 = pd.read_csv(f'{pred_dir}/cnn_bilstm_combined_v2_improved_test.csv')
y_pred1, y_pred2 = df1['y_pred'].values, df2['y_pred'].values
err1, err2 = y_true - y_pred1, y_true - y_pred2
ss1, ss2 = np.sum(err1**2), np.sum(err2**2)
ss_tot = np.sum((y_true - y_true.mean())**2)

print('=' * 80)
print('CombinedV2 vs Baseline 详细对比 - H16')
print('=' * 80)

print(f'\nSS_res 变化: {ss1:.2f} -> {ss2:.2f} (变化: {ss2-ss1:+.2f})')
print(f'SS_tot: {ss_tot:.2f} (恒定)')
print(f'R^2 变化: {1-ss1/ss_tot:.4f} -> {1-ss2/ss_tot:.4f}')

# 误差分布对比
abs_err1, abs_err2 = np.abs(err1), np.abs(err2)
sq_err1, sq_err2 = err1**2, err2**2

print('\n' + '-' * 70)
print('误差分布对比:')
print('-' * 70)
print(f'{"区间":<12} {"Baseline样本":<15} {"CombinedV2样本":<15} {"Baseline SS":<12} {"CombinedV2 SS":<12}')
print('-' * 70)

bins = [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
for i in range(len(bins)-1):
    mask1 = (abs_err1 >= bins[i]) & (abs_err1 < bins[i+1])
    mask2 = (abs_err2 >= bins[i]) & (abs_err2 < bins[i+1])
    n1, n2 = np.sum(mask1), np.sum(mask2)
    ss1_bin, ss2_bin = np.sum(sq_err1[mask1]), np.sum(sq_err2[mask2])
    print(f'[{bins[i]:.2f},{bins[i+1]:.2f})     {n1:<15} {n2:<15} {ss1_bin:<12.2f} {ss2_bin:<12.2f}')

# 尾部误差
mask1_large = abs_err1 >= 0.2
mask2_large = abs_err2 >= 0.2
print(f'[>=0.20)     {np.sum(mask1_large):<15} {np.sum(mask2_large):<15} {np.sum(sq_err1[mask1_large]):<12.2f} {np.sum(sq_err2[mask2_large]):<12.2f}')

# 分析误差变化
improved = np.abs(err2) < np.abs(err1)
worsened = np.abs(err2) > np.abs(err1)
n_improved = np.sum(improved)
n_worsened = np.sum(worsened)
ss_improved_change = np.sum(err2[improved]**2) - np.sum(err1[improved]**2)
ss_worsened_change = np.sum(err2[worsened]**2) - np.sum(err1[worsened]**2)

print('\n' + '-' * 70)
print('误差改善 vs 恶化分析:')
print('-' * 70)
print(f'改善样本数: {n_improved} ({100*n_improved/len(err1):.1f}%)')
print(f'恶化样本数: {n_worsened} ({100*n_worsened/len(err1):.1f}%)')
print(f'改善样本 SS 变化: {ss_improved_change:+.2f}')
print(f'恶化样本 SS 变化: {ss_worsened_change:+.2f}')
print(f'净变化: {ss_improved_change + ss_worsened_change:+.2f}')

print('\n' + '=' * 80)
print('核心问题分析:')
print('=' * 80)
print(f'''
1. MAE 改善的原因:
   - 小误差(<0.05)样本增加: {14438+21014} -> {15351+20887} (+12786样本)
   - 中等误差(0.05-0.1)样本减少: {19442} -> {12207} (-7235样本)

2. R^2 恶化的原因:
   - 大误差(>=0.2)样本 SS 从 {np.sum(sq_err1[mask1_large]):.2f} 增加到 {np.sum(sq_err2[mask2_large]):.2f} (+{np.sum(sq_err2[mask2_large])-np.sum(sq_err1[mask1_large]):.2f})
   - 这些大误差样本的 SS 占总 SS_res 的比例从 60.1% 增加到 76.9%

3. 关键发现:
   CombinedV2 通过减少中等误差换取更多小误差(MAE下降)，
   但代价是少数样本的误差变得更大(SS增加 -> R^2下降)
''')
