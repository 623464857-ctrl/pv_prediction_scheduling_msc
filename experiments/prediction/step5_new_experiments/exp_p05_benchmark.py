"""EXP-P05 推理时间标准化测量。"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def benchmark_forward(
    model: torch.nn.Module,
    sample_input: torch.Tensor,
    *,
    warmup_iters: int = 10,
    repeat_iters: int = 100,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """
    仅测量 model.forward，不含 DataLoader / 反归一化 / 保存。
    返回 ms/sample、samples/s、总时间等。
    """
    device = device or torch.device("cpu")
    model = model.to(device)
    model.eval()
    x = sample_input.to(device)

    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()

        times = []
        n_samples = x.shape[0]
        for _ in range(repeat_iters):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

    arr = np.array(times, dtype=np.float64)
    mean_sec = float(arr.mean())
    std_sec = float(arr.std())
    total_sec = mean_sec * repeat_iters
    ms_per_sample = mean_sec / max(n_samples, 1) * 1000.0
    samples_per_sec = n_samples / max(mean_sec, 1e-9)

    return {
        "batch_size": int(n_samples),
        "warmup_iters": warmup_iters,
        "repeat_iters": repeat_iters,
        "mean_forward_sec": mean_sec,
        "std_forward_sec": std_sec,
        "total_inference_sec": total_sec,
        "ms_per_sample": ms_per_sample,
        "samples_per_sec": samples_per_sec,
        "params": count_parameters(model),
    }
