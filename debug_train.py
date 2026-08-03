import torch
import numpy as np
import sys
sys.path.insert(0, ".")

from experiments.prediction.step5_new_experiments.exp_p06_losses import CombinedV2Loss

np.random.seed(42)
torch.manual_seed(42)

# Load real data
X_train = np.load("data/prediction/step5_new_experiments/samples/h1/X_train_seq.npy")
y_train = np.load("data/prediction/step5_new_experiments/samples/h1/y_train_raw.npy")
y_last = np.load("data/prediction/step5_new_experiments/samples/h1/y_last_train.npy")

from sklearn.preprocessing import StandardScaler
y_res = (y_train - y_last).astype(np.float32)
scaler = StandardScaler()
y_res_scaled = scaler.fit_transform(y_res).astype(np.float32)

print(f"X_train: {X_train.shape}, y_res_scaled: {y_res_scaled.shape}")

# Take a batch
n = 128
indices = np.random.choice(len(X_train), n, replace=False)
x = torch.from_numpy(X_train[indices].astype(np.float32))
y = torch.from_numpy(y_res_scaled[indices])

print(f"Batch: X={x.shape}, y={y.shape}")

# Test CombinedV2
combined = CombinedV2Loss(huber_delta=0.1, smoothness_weight=0.05, sunset_weight=0.1, night_weight=1.5)
loss = combined(y, y, x=x)  # pred=target
print(f"CombinedV2 (pred=target): {loss.item():.6f}")

# Test with small noise
torch.manual_seed(42)
y_noise = y + torch.randn_like(y) * 0.1
loss_noise = combined(y_noise, y, x=x)
print(f"CombinedV2 (pred=target+noise): {loss_noise.item():.6f}")

# MSE for comparison
mse_loss = torch.nn.MSELoss()
mse_noise = mse_loss(y_noise, y)
print(f"MSE (pred=target+noise): {mse_noise.item():.6f}")

# Check night term
diff = y_noise - y  # (128, 1)
target_flat = y.detach().ravel()
q_night = torch.quantile(target_flat, 0.05)
night_mask = y < q_night
print(f"\nNight samples: {night_mask.sum()} / {y.numel()}")
print(f"q_night: {q_night:.3f}")

if night_mask.any():
    night_diff = diff[night_mask]
    night_loss = 1.5 * (night_diff**2).mean()
    print(f"Night diff: mean={night_diff.mean():.3f}, std={night_diff.std():.3f}")
    print(f"Night contribution to loss: {night_loss:.6f}")
