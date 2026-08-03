import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load predictions - P07 MSE no-physics
pred_mse = pd.read_csv("data/prediction/step5_new_experiments/predictions/h1/cnn_lstm_mse_improved_test.csv")
y_true = pred_mse["y_true"].values
y_pred = pred_mse["y_pred"].values

# Load test data for physics
X_test = np.load("data/prediction/step5_new_experiments/samples/h1/X_test_seq.npy")
y_test_raw = np.load("data/prediction/step5_new_experiments/samples/h1/y_test_raw.npy")

# Check: does y_true match?
if not np.allclose(y_true, y_test_raw[:, 0]):
    print("WARNING: y_true mismatch!")

# Metrics without physics
rmse_no = np.sqrt(mean_squared_error(y_true, y_pred))
mae_no = mean_absolute_error(y_true, y_pred)
print(f"Without physics: RMSE={rmse_no:.4f}, MAE={mae_no:.4f}")

# Now apply nighttime_zero_constraint
daylight_flag = X_test[:, -1, 6]
y_pred_zeroed = y_pred.copy()
night_mask = daylight_flag <= 0
y_pred_zeroed[night_mask] = 0.0

rmse_zero = np.sqrt(mean_squared_error(y_true, y_pred_zeroed))
mae_zero = mean_absolute_error(y_true, y_pred_zeroed)
print(f"With nighttime_zero: RMSE={rmse_zero:.4f}, MAE={mae_zero:.4f}")
print(f"  Samples zeroed: {night_mask.sum()} / {len(y_true)} ({night_mask.mean()*100:.1f}%)")

# Apply irradiance upper bound
irradiance = X_test[:, -1, 0]
y_pred_both = y_pred_zeroed.copy()
p_max = np.maximum(irradiance, 0) * 0.9
p_max = np.clip(p_max, 0, 1.1)
y_pred_both = np.clip(y_pred_both, 0, p_max)

rmse_both = np.sqrt(mean_squared_error(y_true, y_pred_both))
mae_both = mean_absolute_error(y_true, y_pred_both)
print(f"With both physics: RMSE={rmse_both:.4f}, MAE={mae_both:.4f}")

# Breakdown
changed = np.abs(y_pred_both - y_pred) > 1e-6
print(f"\nSamples where physics changed prediction: {changed.sum()} ({changed.mean()*100:.1f}%)")
if changed.sum() > 0:
    orig_err = y_true[changed] - y_pred[changed]
    new_err = y_true[changed] - y_pred_both[changed]
    print(f"  Mean |orig error|: {np.abs(orig_err).mean():.4f}")
    print(f"  Mean |new error|: {np.abs(new_err).mean():.4f}")
    improved = np.abs(new_err) < np.abs(orig_err)
    print(f"  Improved: {improved.sum()} / {changed.sum()} ({improved.mean()*100:.1f}%)")
    print(f"  Worse: {(~improved).sum()} / {changed.sum()} ({(~improved).mean()*100:.1f}%)")
