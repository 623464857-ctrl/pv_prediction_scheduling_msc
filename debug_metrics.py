import json

print("=== h1 RMSE Comparison ===\n")

tests = [
    ("Baseline", "h1/residual_optuna_metrics.json"),
    ("asymmetric_mse", "h1/improved_loss_asymmetric_mse_metrics.json"),
    ("huber (new)", "h1/improved_loss_huber_metrics.json"),
    ("combined_v2 (new)", "h1/improved_loss_combined_v2_metrics.json"),
]

for name, relpath in tests:
    path = f"data/prediction/step5_new_experiments/metrics/{relpath}"
    try:
        with open(path) as f:
            data = json.load(f)
        best = min([(k, v) for k, v in data.items() if "error" not in v],
                   key=lambda x: x[1]["RMSE"])
        k, m = best
        print(f"{name}:")
        print(f"  Model: {k}")
        print(f"  RMSE={m['RMSE']:.4f}, nRMSE={m['nRMSE']:.4f}, R2={m['R2']:.4f}")
        seg = m.get("segmented", {})
        pk = seg.get("peak", {})
        print(f"  Peak: MAE={pk.get('MAE','N/A')}, RMSE={pk.get('RMSE','N/A')}, R2={pk.get('R2','N/A')}")
        print()
    except Exception as e:
        print(f"{name}: ERROR - {e}\n")
