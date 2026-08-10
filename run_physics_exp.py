"""临时脚本：运行物理约束实验"""
import sys
import json
import logging
from pathlib import Path

sys.path.insert(0, '.')

from experiments.prediction.step5_new_experiments.exp_p05_common import (
    ensure_dirs, METRICS_DIR, MODELS_DIR, PRED_DIR
)
from experiments.prediction.step5_new_experiments.run_exp_p08_physics import train_physics_model

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s', force=True)
logger = logging.getLogger('physics')

# 加载配置
cfg = json.loads(Path('data/prediction/step5_new_experiments/config/exp_p05_base.json').read_text())
cfg['physics_train'] = {
    'batch_size': 256,
    'lr': 0.001,
    'max_epochs': 50,
    'patience': 8,
    'seed': 42
}

results = {}
for h in [1, 4, 16]:
    print(f'=== Horizon {h} ===')
    ensure_dirs(METRICS_DIR / f'h{h}', METRICS_DIR / f'h{h}', PRED_DIR / f'h{h}')
    m = train_physics_model('cnn_bilstm', h, cfg, logger)
    results[h] = m
    print(f'H{h} RMSE: {m.get("RMSE")}, R2: {m.get("R2")}')

print('=== All Physics Experiments Done ===')
for h, m in results.items():
    print(f'H{h}: RMSE={m.get("RMSE")}, R2={m.get("R2")}')
