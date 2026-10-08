# rl/gigpo_advantage.py
import numpy as np
from collections import defaultdict

GAMMA_DEFAULT = 0.95


def compute_step_returns(steps, gamma=GAMMA_DEFAULT):
    returns = [0.0] * len(steps)
    running = 0.0
    for t in reversed(range(len(steps))):
        running = steps[t]["reward"] + gamma * running
        returns[t] = running
    return returns


def compute_gigpo_advantages(group_steps, final_rewards, micro_weight=1.0, gamma=GAMMA_DEFAULT, eps=1e-4):
    G = len(group_steps)
    rewards = np.array(final_rewards, dtype=np.float32)
    macro_adv = (rewards - rewards.mean()) / (rewards.std() + eps)

    all_returns = [compute_step_returns(steps, gamma) for steps in group_steps]
    buckets = defaultdict(list)
    for i, steps in enumerate(group_steps):
        for t, step in enumerate(steps):
            buckets[step["anchor"]].append((i, t, all_returns[i][t]))

    micro_adv = [[0.0] * len(steps) for steps in group_steps]
    for entries in buckets.values():
        if len(entries) < 2:
            continue
        vals = np.array([e[2] for e in entries], dtype=np.float32)
        mean, std = vals.mean(), vals.std() + eps
        for (i, t, v), norm_v in zip(entries, (vals - mean) / std):
            micro_adv[i][t] = float(norm_v)

    total_adv = [[macro_adv[i] + micro_weight * micro_adv[i][t] for t in range(len(group_steps[i]))]
                 for i in range(G)]
    return total_adv, macro_adv.tolist()
