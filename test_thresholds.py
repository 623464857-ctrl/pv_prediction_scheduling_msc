import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

pred_df = pd.read_csv("data/prediction/step5_new_experiments/predictions/h1/cnn_lstm_mse_improved_test.csv")
y_true = pred_df["y_true"].values
y_pred = pred_df["y_pred"].values

X_test = np.load("data/prediction/step5_new_experiments/samples/h1/X_test_seq.npy")
irradiance = X_test[:, -1, 0]
daylight = X_test[:, -1, 6]

print(f"Baseline (no physics): RMSE={np.sqrt(mean_squared_error(y_true, y_pred)):.4f}\n")

print("Testing different irradiance thresholds:")
for thresh in [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
    pred_test = y_pred.copy()
    day_mask = irradiance > thresh
    p_max = np.maximum(irradiance, 0) * 0.85 * 0.80
    for i in range(len(pred_test)):
        if day_mask[i]:
            pred_test[i] = min(pred_test[i], p_max[i])
    # Also apply nighttime zero
    night_mask = daylight <= 0
    pred_test[night_mask] = 0.0
    rmse = np.sqrt(mean_squared_error(y_true, pred_test))
    print(f"  thresh={thresh:.1f}: day_mask={day_mask.sum():5d}, RMSE={rmse:.4f}")

print(f"\nTesting nighttime zero alone:")
pred_nz = y_pred.copy()
pred_nz[daylight <= 0] = 0.0
rmse_nz = np.sqrt(mean_squared_error(y_true, pred_nz))
print(f"  RMSE={rmse_nz:.4f} (samples zeroed: {(daylight <= 0).sum()})")
