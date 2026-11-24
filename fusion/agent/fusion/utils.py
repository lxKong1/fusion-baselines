"""Trajectory utilities used for offline policy training."""
import random

import numpy as np

def discount_cumsum(x, gamma):
    if x.shape[0] == 0:
        raise ValueError("cannot compute returns-to-go for an empty trajectory")
    disc_cumsum = np.zeros_like(x)
    disc_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        disc_cumsum[t] = x[t] + gamma * disc_cumsum[t+1]
    return disc_cumsum

def _sample_window_start(trajectory_length, context_len):
    if context_len < 1:
        raise ValueError("context_len must be positive")
    if trajectory_length < 1:
        raise ValueError("cannot sample an empty trajectory")
    if trajectory_length <= context_len:
        return 0
    return random.randint(0, trajectory_length - context_len)
