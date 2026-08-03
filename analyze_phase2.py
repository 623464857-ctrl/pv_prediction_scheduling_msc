import json
import numpy as np

print("=== Phase 2 h1 Results Comparison ===\n")

# Load all metrics
files = {
    "Baseline (MSE)": "data/prediction/step5_new_experiments/metrics/h1/residual_optuna_metrics.json",
    "Phase1: asymmetric_mse": "data/prediction/step5_new_experiments/metrics/h1/improved_loss_asymmetric_mse_metrics.json",
    "Phase2: huber": "data/prediction/step5_new_experiments/metrics/h1/improved_loss_huber_metrics.json",
    "Phase2: combined_v2": "data/prediction/step5_new_experiments/metrics/h1/improved_loss_combined_v2_metrics.json",
}

best_overall = {}
best_peak = {}

for name, path in files.items():
    try:
        with open(path) as f:
            data = json.load(f)
    except:
        print(f"{name}: FILE NOT FOUND")
        continue

    best_r = float("inf")
    best_p = float("inf")
    best_r_model = ""
    best_p_model = ""

    for model, m in data.items():
        if "error" in m:
            continue
        r = m["RMSE"]
        p = m.get("segmented", {}).get("peak", {}).get("RMSE", float("inf"))
        if r < best_r:
            best_r = r
            best_r_model = model
        if p < best_p:
            best_p = p
            best_p_model = model

    print(f"{name}")
    print(f"  Best Overall RMSE: {best_r:.4f} ({best_r_model})")
    print(f"  Best Peak RMSE: {best_p:.4f} ({best_p_model})")

    if "Baseline" in name:
        best_overall["baseline"] = best_r
        best_peak["baseline"] = best_p
    elif "huber" in name.lower():
        print(f"  vs Baseline: Overall {best_r/best_overall['baseline']:.2%}, Peak {best_p/best_peak['baseline']:.2%}")
    elif "combined_v2" in name.lower():
        print(f"  vs Baseline: Overall {best_r/best_overall['baseline']:.2%}, Peak {best_p/best_peak['baseline']:.2%}")
    print()
