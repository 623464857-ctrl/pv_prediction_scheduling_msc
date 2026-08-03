import json

print("=== Comparing Baseline vs P07-MSE on h1 ===\n")

# Baseline (P04/P05 residual_optuna - the "true" baseline)
baseline_path = "data/prediction/step5_new_experiments/metrics/h1/residual_optuna_metrics.json"
with open(baseline_path) as f:
    baseline = json.load(f)

# P07 MSE (same data, same loader, same training setup)
p07_mse_path = "data/prediction/step5_new_experiments/metrics/h1/improved_loss_mse_metrics.json"
with open(p07_mse_path) as f:
    p07_mse = json.load(f)

print("Baseline (residual_optuna):")
for k, v in baseline.items():
    print(f"  {k}: RMSE={v['RMSE']:.4f}, nRMSE={v['nRMSE']:.4f}, R2={v['R2']:.4f}")

print("\nP07-MSE:")
for k, v in p07_mse.items():
    print(f"  {k}: RMSE={v['RMSE']:.4f}, nRMSE={v['nRMSE']:.4f}, R2={v['R2']:.4f}")

# Check search strategies
print("\n--- Search strategies ---")
for k, v in baseline.items():
    print(f"  {k}: strategy={v.get('search_strategy')}, params={v.get('params')}")
for k, v in p07_mse.items():
    print(f"  {k}: strategy={v.get('search_strategy')}, params={v.get('params')}")
