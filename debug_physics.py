import numpy as np

# Check daylight_flag distribution in test set
X_test = np.load("data/prediction/step5_new_experiments/samples/h1/X_test_seq.npy")
daylight = X_test[:, -1, 6]  # daylight_flag

print("Daylight flag distribution in test set:")
print(f"  min: {daylight.min():.4f}, max: {daylight.max():.4f}")
print(f"  mean: {daylight.mean():.4f}")
print(f"  % positive (>0): {(daylight > 0).sum() / len(daylight) * 100:.1f}%")
print(f"  % zero: {(daylight == 0).sum() / len(daylight) * 100:.1f}%")
print(f"  % negative (<0): {(daylight < 0).sum() / len(daylight) * 100:.1f}%")

# What percentage of test samples would be zeroed out?
threshold = 0.0
zeroed = (daylight <= threshold).sum() / len(daylight) * 100
print(f"\n  % zeroed (flag <= 0): {zeroed:.1f}%")

# Check irradiance distribution
irradiance = X_test[:, -1, 0]
print(f"\nIrradiance distribution:")
print(f"  min: {irradiance.min():.4f}, max: {irradiance.max():.4f}")
print(f"  % near zero (<0.05): {(irradiance < 0.05).sum() / len(irradiance) * 100:.1f}%")

# Check y_last_test to see what fraction are near 0
y_last = np.load("data/prediction/step5_new_experiments/samples/h1/y_last_test.npy")
print(f"\ny_last_test distribution:")
print(f"  min: {y_last.min():.4f}, max: {y_last.max():.4f}")
print(f"  % zero: {(y_last == 0).sum() / len(y_last) * 100:.1f}%")
print(f"  % near zero (<0.01): {(y_last < 0.01).sum() / len(y_last) * 100:.1f}%")
