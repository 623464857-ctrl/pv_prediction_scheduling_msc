import torch
import numpy as np
import sys
sys.path.insert(0, ".")

from experiments.prediction.step5_new_experiments.exp_p06_losses import CombinedV2Loss

# Load real data
X_train = np.load("data/prediction/step5_new_experiments/samples/h1/X_train_seq.npy")
y_train = np.load("data/prediction/step5_new_experiments/samples/h1/y_train_raw.npy")
y_last = np.load("data/prediction/step5_new_experiments/samples/h1/y_last_train.npy")

# Residual target
y_res = (y_train - y_last).astype(np.float32)

# Scale it like in training
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
y_res_scaled = scaler.fit_transform(y_res).astype(np.float32)

# Take a batch
B = 256
indices = np.random.choice(len(X_train), B, replace=False)
x = torch.from_numpy(X_train[indices].astype(np.float32))
y = torch.from_numpy(y_res_scaled[indices])

# Create loss
combined = CombinedV2Loss(huber_delta=0.1, smoothness_weight=0.05, sunset_weight=0.1, night_weight=3.0)
print(f"Batch size: {B}, Horizon: {y.shape[1]}")
print(f"Target range: [{y.min():.3f}, {y.max():.3f}], mean: {y.mean():.3f}")
print(f"Target std: {y.std():.3f}")

# Test loss
loss = combined(y, y, x=x)  # with perfect predictions, loss should be ~0
print(f"\nCombinedV2 loss (pred=target): {loss.item():.6f}")

# Check night component
diff = y - y
abs_diff = torch.abs(diff)
huber_loss = torch.where(
    abs_diff <= 0.1,
    0.5 * diff**2,
    0.1 * (abs_diff - 0.05),
).mean()
target_flat = y.detach().ravel()
q_night = torch.quantile(target_flat, 0.05)
night_mask = y < q_night
night_diff = diff[night_mask]
if night_mask.any():
    night_loss = 3.0 * (night_diff**2).mean()
else:
    night_loss = 0.0
base = huber_loss + 0.5 * night_loss
print(f"  Base (Huber): {huber_loss.item():.6f}")
print(f"  Night loss: {night_loss:.6f} (samples: {night_mask.sum().item()})")
print(f"  q_night: {q_night:.3f}")
print(f"  Night sample fraction: {night_mask.float().mean():.3f}")

# What if we use MSE instead?
mse_loss = (diff**2).mean()
print(f"\nMSE loss (pred=target): {mse_loss.item():.6f}")
print(f"MSE vs CombinedV2 ratio: {loss.item()/max(mse_loss.item(), 1e-9):.1f}x")

# Compare night-weighted vs raw
print(f"\nWith pred=target, night term dominates:")
print(f"  0.5 * night_loss = {0.5 * night_loss:.6f}")
print(f"  This means ~{(0.5 * night_loss) / max(loss.item(), 1e-9) * 100:.0f}% of loss is from night term")
