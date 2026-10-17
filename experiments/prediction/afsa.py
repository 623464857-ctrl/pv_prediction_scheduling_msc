"""
人工鱼群算法 (Artificial Fish Swarm Algorithm, AFSA) —— 离散超参数版本

每条"鱼"是一组超参数, 用它在每个参数候选列表中的下标表示, 例如
    {"lr": 2, "hidden_size": 1, ...}  表示 lr 取第 3 个候选值, hidden_size 取第 2 个。
"食物浓度"是验证集损失 (越小越好)。

每轮迭代中, 每条鱼依次尝试三种行为, 只要找到更好的位置就移动过去:
    1. 追尾 (follow): 视野内有更好的鱼, 且那里不拥挤 -> 向它游一步
    2. 聚群 (swarm) : 视野内同伴的中心更好, 且不拥挤   -> 向中心游一步
    3. 觅食 (prey)  : 在视野内随机试探 try_number 次, 找到更好的就过去; 都不好就留在原地

评估一组超参数要完整训练一次模型, 代价很高, 所以用 budget 限制"新评估"的次数,
重复出现的超参数组合直接查缓存, 不重复训练。
"""

from __future__ import annotations

from typing import Callable

import numpy as np


class AFSA:
    def __init__(
        self,
        space: dict[str, list],
        objective: Callable[[dict], float],
        n_fish: int = 5,
        visual: float = 0.5,
        crowd: float = 0.6,
        try_number: int = 2,
        seed: int = 42,
    ):
        """
        space:      {参数名: 候选值列表}
        objective:  输入超参数字典, 返回验证损失 (调用方负责缓存)
        visual:     视野半径 (各维下标按 [0,1] 归一化后的欧氏距离)
        crowd:      拥挤度阈值, 视野内同伴占比超过它就不往那里挤
        """
        self.space = space
        self.names = list(space)
        self.sizes = np.array([len(space[k]) for k in self.names])
        self.objective = objective
        self.n_fish, self.visual, self.crowd, self.try_number = n_fish, visual, crowd, try_number
        self.rng = np.random.default_rng(seed)

    # ---- 位置 <-> 超参数 ----------------------------------------------------
    def to_params(self, pos: np.ndarray) -> dict:
        return {k: self.space[k][int(i)] for k, i in zip(self.names, pos)}

    def to_pos(self, params: dict) -> np.ndarray:
        return np.array([self.space[k].index(params[k]) for k in self.names])

    def _dist(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm((a - b) / np.maximum(self.sizes - 1, 1)) / np.sqrt(len(self.names)))

    def _step_towards(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """每个不同的维度朝目标移动 1 个下标。"""
        return a + np.sign(b - a).astype(int)

    def _random_in_visual(self, a: np.ndarray) -> np.ndarray:
        """在视野内随机找一个邻近位置 (随机改 1~2 个维度, 每维 ±1)。"""
        new = a.copy()
        for d in self.rng.choice(len(a), size=self.rng.integers(1, 3), replace=False):
            new[d] = np.clip(new[d] + self.rng.choice([-1, 1]), 0, self.sizes[d] - 1)
        return new

    def _eval(self, pos: np.ndarray) -> float:
        return self.objective(self.to_params(pos))

    # ---- 主循环 ------------------------------------------------------------
    def run(self, init_params: list[dict], n_iter: int, stop: Callable[[], bool] = lambda: False):
        """init_params: 初始鱼群 (例如 Optuna 找到的最好几组)。stop(): 返回 True 时提前结束 (预算用完)。"""
        fish = [self.to_pos(p) for p in init_params[: self.n_fish]]
        while len(fish) < self.n_fish:
            fish.append(np.array([self.rng.integers(0, s) for s in self.sizes]))
        fit = [self._eval(f) for f in fish]

        for _ in range(n_iter):
            for i in range(self.n_fish):
                if stop():
                    break
                fish[i], fit[i] = self._behave(i, fish, fit)
            if stop():
                break
        best = int(np.argmin(fit))
        return self.to_params(fish[best]), fit[best]

    def _behave(self, i: int, fish: list, fit: list):
        me, my_fit = fish[i], fit[i]
        neighbors = [j for j in range(len(fish)) if j != i and self._dist(fish[j], me) <= self.visual]
        not_crowded = len(neighbors) / len(fish) < self.crowd

        # 1. 追尾
        if neighbors and not_crowded:
            j = min(neighbors, key=lambda k: fit[k])
            if fit[j] < my_fit:
                cand = self._step_towards(me, fish[j])
                f = self._eval(cand)
                if f < my_fit:
                    return cand, f

        # 2. 聚群
        if neighbors and not_crowded:
            center = np.round(np.mean([fish[j] for j in neighbors], axis=0)).astype(int)
            cand = self._step_towards(me, center)
            if not np.array_equal(cand, me):
                f = self._eval(cand)
                if f < my_fit:
                    return cand, f

        # 3. 觅食
        for _ in range(self.try_number):
            cand = self._random_in_visual(me)
            f = self._eval(cand)
            if f < my_fit:
                return cand, f
        return me, my_fit  # 没找到更好的, 原地不动
