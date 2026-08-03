import json

print("=" * 70)
print("Phase 1 & Phase 2 Results Comparison - Horizon 1")
print("=" * 70)

tests = [
    ("Baseline (MSE)", "h1/residual_optuna_metrics.json", "mse"),
    ("Phase 1: asymmetric_mse", "h1/improved_loss_asymmetric_mse_metrics.json", "asymmetric_mse"),
    ("Phase 2: huber (delta=0.1)", "h1/improved_loss_huber_metrics.json", "huber"),
    ("Phase 2: combined_v2", "h1/improved_loss_combined_v2_metrics.json", "combined_v2"),
]

results = []
for name, relpath, suffix in tests:
    path = f"data/prediction/step5_new_experiments/metrics/{relpath}"
    try:
        with open(path) as f:
            data = json.load(f)
        entries = [(k, v) for k, v in data.items() if "error" not in v]
        best_rmse = min(entries, key=lambda x: x[1]["RMSE"])
        best_peak_rmse_entry = min(entries, key=lambda x: x[1].get("segmented", {}).get("peak", {}).get("RMSE", float("inf")))
        seg = best_peak_rmse_entry[1].get("segmented", {})
        results.append({
            "name": name,
            "best_rmse": best_rmse[1]["RMSE"],
            "best_rmse_model": best_rmse[0].replace("_improved", "").replace(f"_{suffix}", ""),
            "best_nrmse": best_rmse[1]["nRMSE"],
            "best_r2": best_rmse[1]["R2"],
            "peak_rmse": seg.get("peak", {}).get("RMSE"),
            "peak_r2": seg.get("peak", {}).get("R2"),
            "mid_rmse": seg.get("mid", {}).get("RMSE"),
            "low_rmse": seg.get("low_power", {}).get("RMSE"),
        })
    except Exception as e:
        results.append({"name": name, "error": str(e)})

baseline_rmse = results[0]["best_rmse"]
baseline_peak = results[0]["peak_rmse"]

for r in results:
    if "error" in r:
        print(f"\n{r['name']}: ERROR - {r['error']}")
        continue
    print(f"\n{r['name']}")
    print(f"  Best Model: {r['best_rmse_model']}")
    print(f"  RMSE: {r['best_rmse']:.4f} ({r['best_rmse']/baseline_rmse:.1%} vs baseline)")
    print(f"  nRMSE: {r['best_nrmse']:.4f}, R2: {r['best_r2']:.4f}")
    if r["peak_rmse"] is not None:
        pb = r['peak_rmse'] / baseline_peak if baseline_peak else "N/A"
        pb_str = f"({pb:.1%} vs baseline)" if isinstance(pb, float) else "(baseline N/A)"
        print(f"  Peak RMSE: {r['peak_rmse']:.4f} {pb_str}")
        print(f"  Peak R2: {r['peak_r2']:.4f}")
    if r["mid_rmse"] is not None:
        print(f"  Mid RMSE: {r['mid_rmse']:.4f}")
    if r["low_rmse"] is not None:
        print(f"  Low RMSE: {r['low_rmse']:.4f}")

print("\n" + "=" * 70)
print("Bug Fixes Applied in Phase 2:")
print("=" * 70)
print("1. [CRITICAL] Training clamp removed: 74.5% of scaled residuals < 0")
print("   - P07 clamped pred to [0, inf] during training, destroying gradients")
print("   - Fixed: clamp only in post-processing")
print("2. [CRITICAL] Irradiance upper bound: was clamping 80% of predictions")
print("   - Old: p_max = G * 0.9 * capacity for ALL samples")
print("   - Fixed: only when irradiance > 0.85")
print("3. [CRITICAL] CombinedV2 night_weight: night samples have 2x scaled variance")
print("   - night_weight amplifies noise, breaks training at any positive value")
print("   - Fixed: default night_weight = 0.0")
print("=" * 70)
print("\nKey Findings:")
print("- Huber (delta=0.1) matches MSE - no improvement over baseline")
print("- CombinedV2 (no night_weight) matches MSE - smoothness/sunset terms neutral")
print("- Phase 1 asymmetric_mse was 3.6% worse than baseline")
print("- Phase 2 losses do NOT improve peak or sunset prediction")
print("=" * 70)
