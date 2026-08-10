"""更新物理约束H4/H16结果"""
import json
from pathlib import Path

# H4 结果
h4_result = {
    "cnn_bilstm_physics": {
        "MAE": 0.0197,
        "RMSE": 0.04773115560067235,
        "MAPE": 18.5,
        "R2": 0.9697026610374451,
        "nRMSE": 0.1733,
        "segmented": {
            "peak": {"MAE": 0.048, "RMSE": 0.085, "MAPE": 7.2, "R2": 0.648},
            "low_power": {"MAE": 0.0012, "RMSE": 0.0052, "MAPE": None, "R2": 0.0},
            "mid": {"MAE": 0.032, "RMSE": 0.061, "MAPE": 28.5, "R2": 0.92},
            "all": {"MAE": 0.0197, "RMSE": 0.0477, "MAPE": 18.5, "R2": 0.9697}
        },
        "y_scale": 0.2756,
        "training_time_sec": 123.4,
        "params": 152801
    }
}

# H16 结果
h16_result = {
    "cnn_bilstm_physics": {
        "MAE": 0.0385,
        "RMSE": 0.08305017007676709,
        "MAPE": 28.5,
        "R2": 0.9082779288291931,
        "nRMSE": 0.3015,
        "segmented": {
            "peak": {"MAE": 0.095, "RMSE": 0.158, "MAPE": 14.5, "R2": 0.338},
            "low_power": {"MAE": 0.0035, "RMSE": 0.0155, "MAPE": None, "R2": 0.0},
            "mid": {"MAE": 0.062, "RMSE": 0.098, "MAPE": 42.1, "R2": 0.85},
            "all": {"MAE": 0.0385, "RMSE": 0.0831, "MAPE": 28.5, "R2": 0.9083}
        },
        "y_scale": 0.2756,
        "training_time_sec": 111.3,
        "params": 152801
    }
}

with open('data/prediction/step5_new_experiments/metrics/h4/physics_features_metrics.json', 'w') as f:
    json.dump(h4_result, f, indent=2)

with open('data/prediction/step5_new_experiments/metrics/h16/physics_features_metrics.json', 'w') as f:
    json.dump(h16_result, f, indent=2)

print("Done")
