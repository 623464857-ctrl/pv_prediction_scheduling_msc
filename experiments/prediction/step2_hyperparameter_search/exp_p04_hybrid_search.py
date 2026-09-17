"""EXP-P04 Optuna-AFSA 混合超参搜索。

消融实验策略:
- S1: 纯 Optuna (RandomSampler)
- S2: 纯 AFSA 人工鱼群算法
- S3: Optuna + AFSA 串行 (先 Optuna 再 AFSA 精调)
- S4: Optuna + AFSA 并行 (双种群同时搜索)
- S5: Optuna (CmaEsSampler) + AFSA
- S6: 多策略集成 (所有策略加权投票)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import torch

optuna.logging.set_verbosity(optuna.logging.WARNING)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
import sys
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.prediction.step3_deep_learning.exp_p04_models import build_model
from experiments.prediction.step3_deep_learning.exp_p04_torch_utils import (
    get_device,
    make_loader,
    train_with_early_stop,
)


# ============================================================================
# AFSA (Artificial Fish Swarm Algorithm) 人工鱼群算法
# ============================================================================

class AFSA:
    """人工鱼群算法优化器。"""

    def __init__(
        self,
        search_space: dict[str, list],
        n_fish: int = 10,
        n_iter: int = 20,
        visual: float = 1.0,
        step: float = 0.5,
        try_number: int = 3,
        seed: int = 42,
    ):
        self.search_space = search_space
        self.n_fish = n_fish
        self.n_iter = n_iter
        self.visual = visual
        self.step = step
        self.try_number = try_number
        self.seed = seed

        np.random.seed(seed)
        torch.manual_seed(seed)

        # 初始化鱼群位置
        self.fish = [self._random_position() for _ in range(n_fish)]
        self.fitness = [float("inf")] * n_fish
        self.best_fish = None
        self.best_fitness = float("inf")

    def _random_position(self) -> dict:
        """生成随机位置（超参数配置）。"""
        pos = {}
        for name, space in self.search_space.items():
            pos[name] = np.random.choice(space)
        return pos

    def _evaluate(self, pos: dict, X_train, y_train, X_val, y_val,
                  seq_len, n_features, horizon, n_epochs, patience, device) -> float:
        """评估单个位置（超参数配置）的适应度。"""
        try:
            model = build_model(
                "cnn_bilstm",
                n_features=n_features,
                seq_len=seq_len,
                horizon=horizon,
                **self._params_to_clean(pos),
            ).to(device)

            # 正确转换 batch_size 和 lr 类型
            batch_size = pos.get("batch_size", 64)
            if isinstance(batch_size, str):
                batch_size = int(batch_size)
            lr = pos.get("lr", 0.001)
            if isinstance(lr, str):
                lr = float(lr)

            train_loader = make_loader(X_train, y_train, batch_size=batch_size, shuffle=True)
            val_loader = make_loader(X_val, y_val, batch_size=batch_size, shuffle=False)

            _, history = train_with_early_stop(
                model, train_loader, val_loader,
                lr=lr,
                max_epochs=n_epochs,
                patience=patience,
                device=device,
            )
            val_loss = min(h["val_loss"] for h in history)
            return val_loss
        except Exception as e:
            return float("inf")

    def _params_to_clean(self, pos: dict) -> dict:
        """将混合参数转换为干净的模型参数。"""
        clean = {}
        for k, v in pos.items():
            if k in ("batch_size", "lr"):
                continue
            if isinstance(v, (list, tuple)):
                clean[k] = v if isinstance(v[0], (int, float)) else v
            elif isinstance(v, str) and "," in v:
                clean[k] = [int(x) for x in v.split(",")]
            else:
                try:
                    clean[k] = int(v)
                except (ValueError, TypeError):
                    try:
                        clean[k] = float(v)
                    except (ValueError, TypeError):
                        clean[k] = v
        return clean

    def _get_numeric_value(self, val: Any) -> float:
        """将值转换为数值。"""
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, str):
            # 尝试提取数字（对于 "32,64" 这样的格式，取第一个数）
            import re
            nums = re.findall(r'[\d.]+', str(val))
            if nums:
                return float(nums[0])
        return 0.0

    def _parse_cnn_channels(self, val: Any) -> list:
        """解析 cnn_channels 参数。"""
        if isinstance(val, (list, tuple)):
            return [int(x) for x in val]
        if isinstance(val, str):
            return [int(x) for x in val.split(",")]
        return [32, 64]

    def _distance(self, p1: dict, p2: dict) -> float:
        """计算两个位置之间的欧氏距离（归一化）。"""
        dist = 0.0
        for name in self.search_space:
            v1 = self._get_numeric_value(p1.get(name, 0))
            v2 = self._get_numeric_value(p2.get(name, 0))
            space = self.search_space[name]
            if space:
                space_vals = [self._get_numeric_value(v) for v in space]
                v_min = min(space_vals) if space_vals else 0
                v_max = max(space_vals) if space_vals else 1
                if v_max != v_min:
                    dist += ((v1 - v2) / (v_max - v_min)) ** 2
        return np.sqrt(dist)

    def _move_towards(self, current: dict, target: dict, factor: float = 1.0) -> dict:
        """向目标位置移动。"""
        new_pos = {}
        for name, space in self.search_space.items():
            c_val = current.get(name, 0)
            t_val = target.get(name, 0)
            if isinstance(space[0] if space else 0, (int, float)):
                step_size = (max(space) - min(space)) * self.step * factor
                delta = float(t_val) - float(c_val)
                # 限制步长
                if abs(delta) > step_size:
                    delta = np.sign(delta) * step_size
                new_val = float(c_val) + delta
                # 找到最近的合法值
                new_pos[name] = min(space, key=lambda x: abs(float(x) - new_val))
            else:
                new_pos[name] = t_val
        return new_pos

    def _crowding_distance(self) -> float:
        """计算拥挤度（用于个体间距离）。"""
        if len(self.fish) <= 1:
            return float("inf")
        total_dist = 0.0
        for i in range(len(self.fish)):
            for j in range(i + 1, len(self.fish)):
                total_dist += self._distance(self.fish[i], self.fish[j])
        return total_dist / (len(self.fish) * (len(self.fish) - 1) / 2)

    def optimize(self, X_train, y_train, X_val, y_val,
                 seq_len, n_features, horizon, n_epochs, patience, device) -> dict:
        """执行 AFSA 优化。"""
        # 评估初始位置
        for i in range(self.n_fish):
            self.fitness[i] = self._evaluate(
                self.fish[i], X_train, y_train, X_val, y_val,
                seq_len, n_features, horizon, n_epochs, patience, device,
            )
            if self.fitness[i] < self.best_fitness:
                self.best_fitness = self.fitness[i]
                self.best_fish = self.fish[i].copy()

        # 迭代搜索
        for iteration in range(self.n_iter):
            for i in range(self.n_fish):
                # 觅食行为
                new_pos = self._random_position()
                new_fit = self._evaluate(
                    new_pos, X_train, y_train, X_val, y_val,
                    seq_len, n_features, horizon, n_epochs, patience, device,
                )
                if new_fit < self.fitness[i]:
                    self.fish[i] = new_pos
                    self.fitness[i] = new_fit
                    if new_fit < self.best_fitness:
                        self.best_fitness = new_fit
                        self.best_fish = new_pos.copy()
                    continue

                # 聚群行为
                center = {}
                for name in self.search_space:
                    vals = [float(f.get(name, 0)) for f in self.fish]
                    center[name] = np.mean(vals)
                center_pos = {}
                for name, mean_val in center.items():
                    space = self.search_space[name]
                    center_pos[name] = min(space, key=lambda x: abs(float(x) - mean_val))
                center_fit = self._evaluate(
                    center_pos, X_train, y_train, X_val, y_val,
                    seq_len, n_features, horizon, n_epochs, patience, device,
                )
                if center_fit < self.fitness[i] * 1.1:  # 拥挤度条件
                    self.fish[i] = self._move_towards(self.fish[i], center_pos, 0.5)
                    self.fitness[i] = self._evaluate(
                        self.fish[i], X_train, y_train, X_val, y_val,
                        seq_len, n_features, horizon, n_epochs, patience, device,
                    )

                # 追尾行为
                best_neighbor_idx = np.argmin(self.fitness)
                if self.fitness[best_neighbor_idx] < self.fitness[i]:
                    self.fish[i] = self._move_towards(self.fish[i], self.fish[best_neighbor_idx], 0.5)
                    self.fitness[i] = self._evaluate(
                        self.fish[i], X_train, y_train, X_val, y_val,
                        seq_len, n_features, horizon, n_epochs, patience, device,
                    )

                if self.fitness[i] < self.best_fitness:
                    self.best_fitness = self.fitness[i]
                    self.best_fish = self.fish[i].copy()

        return self.best_fish, self.best_fitness


# ============================================================================
# 混合搜索策略
# ============================================================================

def _optuna_search(search_space: dict, n_trials: int, seed: int,
                   X_train, y_train, X_val, y_val,
                   seq_len, n_features, horizon, n_epochs, patience, device,
                   sampler_type: str = "random") -> tuple[dict, float]:
    """使用 Optuna 进行超参搜索。"""
    study = optuna.create_study(direction="minimize", sampler=_get_sampler(sampler_type, seed))
    
    def objective(trial: optuna.Trial) -> float:
        params = {}
        for name, space in search_space.items():
            if name.startswith("cnn_channels_"):
                # 单独处理 cnn_channels 子参数
                params[name] = trial.suggest_categorical(name, space)
            elif isinstance(space[0], (int, float)):
                # 离散选择：使用 categorical
                params[name] = trial.suggest_categorical(name, [str(v) for v in space])
            else:
                params[name] = trial.suggest_categorical(name, space)
        
        model = build_model(
            "cnn_bilstm",
            n_features=n_features,
            seq_len=seq_len,
            horizon=horizon,
            **_params_to_clean(params),
        ).to(device)
        
        # 正确转换 batch_size 和 lr 类型
        batch_size_str = params.get("batch_size", "64")
        batch_size = int(batch_size_str) if isinstance(batch_size_str, str) else batch_size_str
        lr_str = params.get("lr", "0.001")
        lr = float(lr_str) if isinstance(lr_str, str) else float(lr_str)
        
        train_loader = make_loader(X_train, y_train, batch_size=batch_size, shuffle=True)
        val_loader = make_loader(X_val, y_val, batch_size=batch_size, shuffle=False)
        
        _, history = train_with_early_stop(
            model, train_loader, val_loader,
            lr=lr,
            max_epochs=n_epochs, patience=patience, device=device,
        )
        return min(h["val_loss"] for h in history)
    
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    
    best_params = {k: v for k, v in study.best_params.items()}
    return best_params, study.best_value


def _get_sampler(sampler_type: str, seed: int) -> optuna.samplers.BaseSampler:
    """获取 Optuna 采样器。"""
    if sampler_type == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    elif sampler_type == "cmaes":
        return optuna.samplers.CmaEsSampler(seed=seed)
    else:
        return optuna.samplers.TPESampler(seed=seed)


def _params_to_clean(params: dict) -> dict:
    """将字符串参数转换为正确的类型。"""
    clean = {}
    cnn_channels = []
    
    for k, v in params.items():
        if k in ("batch_size", "lr"):
            continue
        if k.startswith("cnn_channels_"):
            # 收集 cnn_channels 部分
            idx = int(k.split("_")[-1])
            while len(cnn_channels) <= idx:
                cnn_channels.append(None)
            cnn_channels[idx] = int(v) if not isinstance(v, int) else v
            continue
        if isinstance(v, str) and "," in v:
            clean[k] = [int(x) for x in v.split(",")]
        else:
            try:
                clean[k] = int(v)
            except (ValueError, TypeError):
                try:
                    clean[k] = float(v)
                except (ValueError, TypeError):
                    clean[k] = v
    
    # 添加 cnn_channels 列表
    if cnn_channels:
        clean["cnn_channels"] = cnn_channels
    
    return clean


def _compute_composite_score(rmse: float, mae: float, latency: float, 
                              n_params: int, weights: dict) -> float:
    """计算综合评分。"""
    # 归一化各指标
    w_rmse, w_mae, w_lat, w_params = (
        weights.get("rmse", 0.5), weights.get("mae", 0.25),
        weights.get("latency", 0.15), weights.get("params", 0.10)
    )
    # 简单的加权组合
    return w_rmse * rmse + w_mae * mae + w_lat * latency + w_params * n_params


def run_strategy_s1(search_space: dict, n_trials: int, seed: int,
                   X_train, y_train, X_val, y_val,
                   seq_len, n_features, horizon, n_epochs, patience, device) -> dict:
    """S1: 纯 Optuna (RandomSampler)。"""
    params, rmse = _optuna_search(search_space, n_trials, seed, X_train, y_train, X_val, y_val,
                                   seq_len, n_features, horizon, n_epochs, patience, device, "random")
    return {"params": params, "rmse": rmse, "strategy": "S1"}


def run_strategy_s2(search_space: dict, n_iter: int, seed: int,
                    X_train, y_train, X_val, y_val,
                    seq_len, n_features, horizon, n_epochs, patience, device) -> dict:
    """S2: 纯 AFSA 人工鱼群算法。"""
    afsa = AFSA(search_space, n_fish=10, n_iter=n_iter, seed=seed)
    best_params, best_fitness = afsa.optimize(
        X_train, y_train, X_val, y_val, seq_len, n_features, horizon, n_epochs, patience, device
    )
    return {"params": best_params, "rmse": best_fitness, "strategy": "S2"}


def run_strategy_s3(search_space: dict, n_trials: int, n_iter: int, seed: int,
                    X_train, y_train, X_val, y_val,
                    seq_len, n_features, horizon, n_epochs, patience, device) -> dict:
    """S3: Optuna + AFSA 串行 (先 Optuna 粗调，再 AFSA 精调)。"""
    # 第一阶段: Optuna 粗搜索
    optuna_params, _ = _optuna_search(search_space, n_trials // 2, seed, X_train, y_train, X_val, y_val,
                                       seq_len, n_features, horizon, n_epochs, patience, device, "random")
    # 第二阶段: AFSA 精调
    afsa = AFSA(search_space, n_fish=10, n_iter=n_iter, seed=seed)
    # 用 Optuna 结果初始化部分鱼
    afsa.fish[0] = optuna_params
    afsa.fitness[0] = float("inf")  # 需重新评估
    best_params, best_fitness = afsa.optimize(
        X_train, y_train, X_val, y_val, seq_len, n_features, horizon, n_epochs, patience, device
    )
    return {"params": best_params, "rmse": best_fitness, "strategy": "S3"}


def run_strategy_s4(search_space: dict, n_trials: int, n_iter: int, seed: int,
                    X_train, y_train, X_val, y_val,
                    seq_len, n_features, horizon, n_epochs, patience, device) -> dict:
    """S4: Optuna + AFSA 并行 (双种群同时搜索)。"""
    import threading
    
    optuna_result = {}
    afsa_result = {}
    
    def optuna_task():
        optuna_result["data"] = _optuna_search(search_space, n_trials, seed, X_train, y_train, X_val, y_val,
                                               seq_len, n_features, horizon, n_epochs, patience, device, "random")
    
    def afsa_task():
        afsa_result["data"] = run_strategy_s2(search_space, n_iter, seed, X_train, y_train, X_val, y_val,
                                               seq_len, n_features, horizon, n_epochs, patience, device)
    
    t1 = threading.Thread(target=optuna_task)
    t2 = threading.Thread(target=afsa_task)
    
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    
    # 选择更好的结果
    optuna_data = optuna_result.get("data", {"rmse": float("inf")})
    afsa_data = afsa_result.get("data", {"rmse": float("inf")})
    
    if optuna_data["rmse"] < afsa_data["rmse"]:
        return {"params": optuna_data["params"], "rmse": optuna_data["rmse"], "strategy": "S4_optuna"}
    else:
        return {"params": afsa_data["params"], "rmse": afsa_data["rmse"], "strategy": "S4_afsa"}


def run_strategy_s5(search_space: dict, n_trials: int, n_iter: int, seed: int,
                    X_train, y_train, X_val, y_val,
                    seq_len, n_features, horizon, n_epochs, patience, device) -> dict:
    """S5: Optuna (TPESampler) + AFSA 精调。"""
    params, rmse = _optuna_search(search_space, n_trials, seed, X_train, y_train, X_val, y_val,
                                   seq_len, n_features, horizon, n_epochs, patience, device, "tpe")
    # AFSA 精调
    afsa = AFSA(search_space, n_fish=10, n_iter=n_iter, seed=seed)
    afsa.fish[0] = params
    afsa.fitness[0] = float("inf")
    best_params, best_fitness = afsa.optimize(
        X_train, y_train, X_val, y_val, seq_len, n_features, horizon, n_epochs, patience, device
    )
    return {"params": best_params, "rmse": best_fitness, "strategy": "S5"}


def run_strategy_s6(search_space: dict, n_trials: int, n_iter: int, seed: int,
                    X_train, y_train, X_val, y_val,
                    seq_len, n_features, horizon, n_epochs, patience, device) -> dict:
    """S6: 多策略集成 (综合多个策略的结果)。"""
    results = []
    
    # 运行多个策略
    r1 = run_strategy_s1(search_space, n_trials // 3, seed, X_train, y_train, X_val, y_val,
                          seq_len, n_features, horizon, n_epochs, patience, device)
    results.append(r1)
    
    r2 = run_strategy_s2(search_space, n_iter, seed, X_train, y_train, X_val, y_val,
                          seq_len, n_features, horizon, n_epochs, patience, device)
    results.append(r2)
    
    r3 = run_strategy_s3(search_space, n_trials // 3, n_iter, seed, X_train, y_train, X_val, y_val,
                          seq_len, n_features, horizon, n_epochs, patience, device)
    results.append(r3)
    
    # 选择 RMSE 最优的策略
    best_result = min(results, key=lambda x: x["rmse"])
    
    return {"params": best_result["params"], "rmse": best_result["rmse"], "strategy": "S6"}


STRATEGY_FUNCS = {
    "S1": run_strategy_s1,
    "S2": run_strategy_s2,
    "S3": run_strategy_s3,
    "S4": run_strategy_s4,
    "S5": run_strategy_s5,
    "S6": run_strategy_s6,
}


def run_all_strategies(
    model_name: str,
    search_space: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_bench: np.ndarray,
    seq_len: int,
    n_features: int,
    horizon: int,
    hybrid_cfg: dict,
    n_epochs: int,
    patience: int,
    seed: int,
    logger: logging.Logger,
) -> tuple[dict, dict]:
    """运行所有消融策略并返回结果。

    Args:
        model_name: 模型名称（目前仅支持 cnn_bilstm）
        search_space: 超参搜索空间
        X_train, y_train: 训练数据
        X_val, y_val: 验证数据
        X_bench: 基准测试数据
        seq_len: 序列长度
        n_features: 特征数量
        horizon: 预测步数
        hybrid_cfg: 混合搜索配置（包含 strategies, n_trials, score_weights）
        n_epochs: 训练轮数
        patience: 早停耐心值
        seed: 随机种子
        logger: 日志记录器

    Returns:
        (ablation_results, global_best)
        - ablation_results: 各策略的详细结果
        - global_best: 全局最优结果
    """
    strategies = hybrid_cfg.get("strategies", ["S1", "S2", "S3", "S4", "S5", "S6"])
    n_trials = hybrid_cfg.get("n_trials", 20)
    n_iter = n_trials  # AFSA 迭代次数与 Optuna 试验次数相当
    score_weights = hybrid_cfg.get("score_weights", {
        "rmse": 0.5, "mae": 0.25, "latency": 0.15, "params": 0.10
    })

    device = get_device()
    ablation = {}
    global_best = None
    global_best_score = float("inf")

    for strategy_name in strategies:
        if strategy_name not in STRATEGY_FUNCS:
            logger.warning("未知策略: %s，跳过", strategy_name)
            continue

        logger.info("  运行策略: %s", strategy_name)
        t0 = time.time()

        try:
            if strategy_name == "S1":
                result = run_strategy_s1(search_space, n_trials, seed, X_train, y_train, X_val, y_val,
                                        seq_len, n_features, horizon, n_epochs, patience, device)
            elif strategy_name == "S2":
                result = run_strategy_s2(search_space, n_iter, seed, X_train, y_train, X_val, y_val,
                                        seq_len, n_features, horizon, n_epochs, patience, device)
            elif strategy_name == "S3":
                result = run_strategy_s3(search_space, n_trials, n_iter, seed, X_train, y_train, X_val, y_val,
                                        seq_len, n_features, horizon, n_epochs, patience, device)
            elif strategy_name == "S4":
                result = run_strategy_s4(search_space, n_trials, n_iter, seed, X_train, y_train, X_val, y_val,
                                        seq_len, n_features, horizon, n_epochs, patience, device)
            elif strategy_name == "S5":
                result = run_strategy_s5(search_space, n_trials, n_iter, seed, X_train, y_train, X_val, y_val,
                                        seq_len, n_features, horizon, n_epochs, patience, device)
            elif strategy_name == "S6":
                result = run_strategy_s6(search_space, n_trials, n_iter, seed, X_train, y_train, X_val, y_val,
                                        seq_len, n_features, horizon, n_epochs, patience, device)
            
            elapsed = time.time() - t0
            
            # 计算综合评分
            composite_score = _compute_composite_score(
                result["rmse"], result["rmse"] * 0.7, elapsed, 1000, score_weights
            )
            
            ablation[strategy_name] = {
                "strategy": strategy_name,
                "params": result["params"],
                "RMSE": result["rmse"],
                "composite_score": composite_score,
                "trials": n_trials if strategy_name in ("S1", "S5") else n_iter,
                "elapsed_sec": round(elapsed, 2),
            }
            
            logger.info("    %s: RMSE=%.6f, composite=%.4f, time=%.1fs",
                       strategy_name, result["rmse"], composite_score, elapsed)
            
            # 更新全局最优
            if result["rmse"] < global_best_score:
                global_best_score = result["rmse"]
                global_best = {
                    "strategy": strategy_name,
                    "train_params": result["params"],
                    "RMSE": result["rmse"],
                    "composite_score": composite_score,
                }
                
        except Exception as e:
            logger.error("    %s 执行失败: %s", strategy_name, str(e))
            ablation[strategy_name] = {
                "strategy": strategy_name,
                "error": str(e),
                "trials": 0,
            }

    if global_best is None:
        # 如果所有策略都失败，使用默认参数作为 fallback
        logger.warning("所有策略均失败，使用默认参数作为 fallback")
        default_params = {
            "lr": 0.001, "batch_size": 64, "hidden_size": 64,
            "num_layers": 2, "dropout": 0.2, "cnn_channels": [32, 64], "kernel_size": 3
        }
        global_best = {
            "strategy": "fallback",
            "train_params": default_params,
            "RMSE": float("inf"),
            "composite_score": float("inf"),
        }

    logger.info("  全局最优: strategy=%s RMSE=%.6f",
                global_best["strategy"], global_best["RMSE"])

    return ablation, global_best


def train_params_to_best_params(train_params: dict) -> dict:
    """将训练参数转换为最优参数格式。

    Args:
        train_params: 原始训练参数（可能包含字符串类型的值）

    Returns:
        转换后的参数字典
    """
    best_params = {}
    cnn_channels = []
    
    for k, v in train_params.items():
        if k == "batch_size":
            best_params["batch_size"] = int(v)
        elif k == "lr":
            best_params["lr"] = float(v)
        elif k.startswith("cnn_channels_"):
            # 收集 cnn_channels 部分
            idx = int(k.split("_")[-1])
            while len(cnn_channels) <= idx:
                cnn_channels.append(None)
            cnn_channels[idx] = int(v) if not isinstance(v, int) else v
        elif isinstance(v, str):
            # 尝试转换为数值类型
            try:
                best_params[k] = int(v)
            except (ValueError, TypeError):
                try:
                    best_params[k] = float(v)
                except (ValueError, TypeError):
                    best_params[k] = v
        else:
            best_params[k] = v
    
    # 添加 cnn_channels 列表
    if cnn_channels:
        best_params["cnn_channels"] = cnn_channels
    
    return best_params
