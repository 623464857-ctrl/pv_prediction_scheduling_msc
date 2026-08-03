import json, os

path = "data/prediction/step5_new_experiments/metrics/h1/improved_loss_mse_metrics.json"
with open(path) as f:
    data = json.load(f)
best = min([(k, v) for k, v in data.items() if "error" not in v], key=lambda x: x[1]["RMSE"])
k, m = best
print(f"Model: {k}")
print(f"RMSE: {m['RMSE']:.4f}, R2: {m['R2']:.4f}")
print(f"Search strategy: {m.get('search_strategy')}")
print(f"Training time: {m.get('training_time_sec')}s")
print(f"File mtime: {os.path.getmtime(path)}")

model_path = f"data/prediction/step5_new_experiments/models/h1/cnn_lstm_mse_improved.pt"
if os.path.exists(model_path):
    print(f"Model mtime: {os.path.getmtime(model_path)}")
else:
    print("Model file not found")

# Check the log file
log_path = "logs/prediction/step5_new_experiments/EXP-P07_h1_mse.log"
if os.path.exists(log_path):
    print(f"Log mtime: {os.path.getmtime(log_path)}")
