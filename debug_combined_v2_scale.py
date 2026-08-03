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

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
y_res_scaled = scaler.fit_transform(y_res).astype(np.float32)

# Take a batch
B = 256
np.random.seed(42)
indices = np.random.choice(len(X_train), B, replace=False)
x = torch.from_numpy(X_train[indices].astype(np.float32))
y = torch.from_numpy(y_res_scaled[indices])

# Compute CombinedV2 loss
combined = CombinedV2Loss(huber_delta=0.1, smoothness_weight=0.05, sunset_weight=0.1, night_weight=3.0)
loss = combined(y, y, x=x)
print(f"CombinedV2 (pred=target): {loss.item():.6f}")

# Now simulate: what if model predicts y*0.9?
loss_09 = combined(y * 0.9, y, x=x)
print(f"CombinedV2 (pred=0.9*target): {loss_09.item():.6f}")

# What if model predicts y + noise?
torch.manual_seed(42)
y_noise = y + torch.randn_like(y) * 0.1
loss_noise = combined(y_noise, y, x=x)
print(f"CombinedV2 (pred=target+noise): {loss_noise.item():.6f}")

# What if model predicts near-zero (safe for night, bad for day)?
y_safe = y.clone()
y_safe[y > 0] = y[y > 0] * 0.1  # reduce positive residuals
y_safe[y < 0] = y[y < 0] * 1.1  # keep negative residuals (night)
loss_safe = combined(y_safe, y, x=x)
print(f"CombinedV2 (pred=safe): {loss_safe.item():.6f}")

# Compare with MSE
mse = ((y * 0.9 - y) ** 2).mean()
print(f"\nMSE (pred=0.9*target): {mse.item():.6f}")
mse_noise = ((y_noise - y) ** 2).mean()
print(f"MSE (pred=target+noise): {mse_noise.item():.6f}")

# What is the night term contribution for each?
print("\n--- Night term contribution ---")
for name, y_pred_val in [("target", y), ("0.9*target", y * 0.9), ("noisy", y_noise), ("safe", y_safe)]:
    diff = y_pred_val - y
    target_flat = y.detach().ravel()
    q_night = torch.quantile(target_flat, 0.05)
    night_mask = y < q_night
    if night_mask.any():
        night_loss = 3.0 * (diff[night_mask]**2).mean()
        base_loss = (diff**2).mean()
        print(f"  {name}: night_contrib={0.5*night_loss:.6f}, total_loss={base_loss + 0.5*night_loss:.6f}, night_samples={night_mask.sum()}")
