import numpy as np
from sklearn.metrics import mean_squared_error

# Load predictions and test data
X_test = np.load("data/prediction/step5_new_experiments/samples/h1/X_test_seq.npy")
y_test_raw = np.load("data/prediction/step5_new_experiments/samples/h1/y_test_raw.npy")
y_true = y_test_raw[:, 0]

irradiance = X_test[:, -1, 0]
capacity = 1.0

# Current formula: p_max = max(irradiance, 0) * 0.9 * capacity, clipped to [0, 1.1]
p_max = np.maximum(irradiance, 0) * 0.9 * capacity
p_max = np.clip(p_max, 0, capacity * 1.1)

print("Irradiance upper bound analysis:")
print(f"  Irradiance range: [{irradiance.min():.4f}, {irradiance.max():.4f}]")
print(f"  P_max range: [{p_max.min():.4f}, {p_max.max():.4f}]")
print(f"  Samples with p_max < 1.0: {(p_max < 1.0).sum()} / {len(p_max)} ({(p_max < 1.0).mean()*100:.1f}%)")
print(f"  Samples with p_max < 0.5: {(p_max < 0.5).sum()} / {len(p_max)} ({(p_max < 0.5).mean()*100:.1f}%)")

# What fraction of test data has actual power > p_max?
# i.e., where is this constraint actually clamping?
y_pred = np.load("data/prediction/step5_new_experiments/predictions/h1/cnn_lstm_mse_improved_test.csv")["y_pred"].values
above_bound = y_pred > p_max
print(f"\n  Samples where pred > p_max: {above_bound.sum()} / {len(y_pred)} ({above_bound.mean()*100:.1f}%)")

# What is the real power vs irradiance relationship?
# For samples with irradiance > 0, what's the ratio?
daylight = X_test[:, -1, 6]
day_mask = daylight > 0
if day_mask.sum() > 0:
    # y_true / irradiance for actual relationship
    valid = day_mask & (irradiance > 0.1)
    if valid.sum() > 0:
        ratio = y_true[valid] / irradiance[valid]
        print(f"\n  y_true / irradiance ratio (for irradiance > 0.1):")
        print(f"    Mean: {ratio.mean():.4f}, Std: {ratio.std():.4f}")
        print(f"    Median: {np.median(ratio):.4f}")
        print(f"    90th percentile: {np.percentile(ratio, 90):.4f}")
