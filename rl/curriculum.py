# rl/curriculum.py
import numpy as np


class CurriculumSampler:
    """Приоритизирует задачи с pass-rate в диапазоне [lo, hi] -- максимум дисперсии
    внутри группы G, максимум полезного градиента для GiGPO."""

    def __init__(self, num_tasks, lo=0.15, hi=0.85, ema_alpha=0.2, explore_prob=0.15):
        self.num_tasks = num_tasks
        self.pass_rate = np.full(num_tasks, 0.5, dtype=np.float32)
        self.visits = np.zeros(num_tasks, dtype=np.int32)
        self.lo, self.hi = lo, hi
        self.ema_alpha = ema_alpha
        self.explore_prob = explore_prob

    def update(self, task_idx, rewards: list):
        mean_r = float(np.mean(rewards))
        if self.visits[task_idx] == 0:
            self.pass_rate[task_idx] = mean_r
        else:
            self.pass_rate[task_idx] = (
                (1 - self.ema_alpha) * self.pass_rate[task_idx] + self.ema_alpha * mean_r
            )
        self.visits[task_idx] += 1

    def sample(self) -> int:
        if np.random.rand() < self.explore_prob or np.min(self.visits) == 0:
            unvisited = np.where(self.visits == 0)[0]
            if len(unvisited) > 0:
                return int(np.random.choice(unvisited))
            return int(np.random.randint(self.num_tasks))

        in_range = np.where((self.pass_rate >= self.lo) & (self.pass_rate <= self.hi))[0]
        if len(in_range) == 0:
            return int(np.random.randint(self.num_tasks))
        weights = 1.0 - np.abs(self.pass_rate[in_range] - 0.5) * 2.0
        weights = np.clip(weights, 1e-3, None)
        weights /= weights.sum()
        return int(np.random.choice(in_range, p=weights))

    def mean_pass_rate(self):
        visited = self.visits > 0
        return float(self.pass_rate[visited].mean()) if visited.any() else float("nan")
