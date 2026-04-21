from __future__ import annotations

import numpy as np


class RandomAgent:
    def __init__(self, action_space, seed: int | None = None):
        self.action_space = action_space
        if seed is not None:
            self.action_space.seed(seed)

    def act(self, observation) -> int:
        return int(self.action_space.sample())
