import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

# Load predictions from the no-physics run
pred = pd.read_csv("data/prediction/step5_new_experiments/predictions/h1/cnn_lstm_mse_improved_test.csv")
y_true = pred["y_true"].values
y_pred = pred["y_pred"].values

# Load test data
X_test = np.load("data/prediction/step5_new_experiments/samples/h1/X_test_seq.npy")
irradiance = X_test[:, -1, 0]
daylight = X_test[:, -1, 6]

print("=== Testing irradiance_upper_bound fix ===\n")
print(f"Baseline (no physics): RMSE={np.sqrt(mean_squared_error(y_true, y_pred)):.4f}")

# OLD formula
pred_old = y_pred.copy()
irradiance_expanded_old = np.repeat(irradiance, 1)
p_max_old = np.maximum(irradiance_expanded_old, 0) * 0.9
p_max_old = np.clip(p_max_old, 0, 1.1)
np.clip(pred_old, 0, p_max_old, out=pred_old)
print(f"OLD irradiance_bound: RMSE={np.sqrt(mean_squared_error(y_true, pred_old)):.4f}")
print(f"  Samples clamped: {(pred_old != y_pred).sum()}")

# NEW formula - only daytime (>0.3)
pred_new = y_pred.copy()
day_mask = irradiance > 0.3
p_max_new = np.maximum(irradiance, 0) * 0.85 * 0.80  # efficiency * fill_factor
for i in range(len(pred_new)):
    if day_mask[i]:
        pred_new[i] = min(pred_new[i], p_max_new[i])
print(f"NEW irradiance_bound (day only, threshold=0.3): RMSE={np.sqrt(mean_squared_error(y_true, pred_new)):.4f}")
print(f"  Samples affected: {day_mask.sum()} / {len(pred_new)} ({day_mask.mean()*100:.1f}%)")

# Also test with nighttime zero
pred_both = pred_new.copy()
night_mask = daylight <= 0
pred_both[night_mask] = 0.0
print(f"NEW + nighttime_zero: RMSE={np.sqrt(mean_squared_error(y_true, pred_both)):.4f}")
