import torch
import numpy as np
import sys
sys.path.insert(0, ".")

from experiments.prediction.step5_new_experiments.exp_p06_losses import CombinedV2Loss, HuberLoss

# Create a small test
torch.manual_seed(42)
np.random.seed(42)

# Simulate batch data
B, H = 32, 1
pred = torch.randn(B, H) * 0.1  # small values (residual scale)
target = torch.randn(B, H) * 0.1

# Mock x with features
F = 27
x = torch.randn(B, 16, F)

# Test Huber loss
huber = HuberLoss(delta=0.1)
h_loss = huber(pred, target)
print(f"Huber loss (delta=0.1): {h_loss.item():.6f}")

# Test CombinedV2
combined = CombinedV2Loss(huber_delta=0.1, smoothness_weight=0.05, sunset_weight=0.1, night_weight=3.0)
c_loss = combined(pred, target, x=x)
print(f"CombinedV2 loss: {c_loss.item():.6f}")

# Break down
diff = pred - target
abs_diff = torch.abs(diff)
huber_loss = torch.where(
    abs_diff <= 0.1,
    0.5 * diff**2,
    0.1 * (abs_diff - 0.05),
)
base = huber_loss.mean()
print(f"  Base (Huber): {base.item():.6f}")

# Night loss
target_flat = target.detach().ravel()
q_night = torch.quantile(target_flat, 0.05)
night_mask = target < q_night
if night_mask.any():
    night_loss = (diff[night_mask]**2).mean()
    print(f"  Night loss: {night_loss.item():.6f} (samples: {night_mask.sum().item()})")
    print(f"  q_night: {q_night.item():.6f}")

# Smoothness
smooth = torch.tensor(0.0)
if pred.shape[1] > 1:
    smooth = torch.nn.functional.mse_loss(pred[:, 1:], pred[:, :-1])
print(f"  Smooth loss: {smooth.item():.6f}")

# Sunset
sunset_loss = torch.tensor(0.0)
if x is not None and x.shape[-1] > 8 and pred.shape[1] > 1:
    cos_hour = x[:, -1, 8]
    irradiance = x[:, -1, 0]
    sunset_mask = (cos_hour < 0) & (irradiance < 0.4) & (irradiance > 0.05)
    if sunset_mask.any():
        diff_h = pred[:, 1:] - pred[:, :-1]
        positive_grad = torch.clamp(diff_h, min=0.0)
        sunset_loss = positive_grad[sunset_mask].mean()
print(f"  Sunset loss: {sunset_loss.item():.6f}")

print(f"\nTotal (base + 0.5*night + 0.05*smooth + 0.1*sunset): {base + 0.5*night_loss + 0.05*smooth + 0.1*sunset_loss:.6f}")
