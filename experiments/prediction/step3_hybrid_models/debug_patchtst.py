"""调试 PatchTST 模型。"""
import torch
from exp_p03_models import PatchTSTRegressor

model = PatchTSTRegressor(seq_len=16, n_features=13, patch_len=4, stride=2, d_model=64, n_heads=4, num_layers=2, dropout=0.2)
x = torch.randn(2, 16, 13)
print("Input shape:", x.shape)
out = model(x)
print("Output shape:", out.shape)
print("OK")
