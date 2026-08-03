import numpy as np

y_train_raw = np.load("data/prediction/step5_new_experiments/samples/h1/y_train_raw.npy")
y_last_train = np.load("data/prediction/step5_new_experiments/samples/h1/y_last_train.npy")
residuals = (y_train_raw - y_last_train).astype(np.float32)

print(f"Residual stats:")
print(f"  shape: {residuals.shape}")
print(f"  min: {residuals.min():.4f}, max: {residuals.max():.4f}")
print(f"  mean: {residuals.mean():.4f}, std: {residuals.std():.4f}")
print(f"  % negative: {(residuals < 0).sum() / residuals.size * 100:.1f}%")
print(f"  % positive: {(residuals > 0).sum() / residuals.size * 100:.1f}%")
print(f"  % zero: {(residuals == 0).sum() / residuals.size * 100:.1f}%")

# After scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(residuals)
res_scaled = scaler.transform(residuals)
print(f"\nAfter scaling:")
print(f"  min: {res_scaled.min():.4f}, max: {res_scaled.max():.4f}")
print(f"  mean: {res_scaled.mean():.6f}, std: {res_scaled.std():.6f}")
print(f"  % negative: {(res_scaled < 0).sum() / res_scaled.size * 100:.1f}%")
