"""Scripted partner policies for solo Overcooked training.

All partners conform to a tiny protocol:
    ``reset()``  — start a new episode
    ``act(obs)`` → int action  — produce an action given partner-side obs

The learner code never inspects partner internals, so we can mix scripted
and trained partners interchangeably.
"""
from __future__ import annotations

import random
from typing import Protocol

import numpy as np

# Overcooked action ids (Action.ALL_ACTIONS in overcooked_ai_py):
#   0: (0,-1)  NORTH      1: (0,1)  SOUTH      2: (1,0)  EAST
#   3: (-1,0)  WEST       4: (0,0)  STAY       5: "interact"
N_ACTIONS = 6


class Partner(Protocol):
    name: str
    def reset(self) -> None: ...
    def act(self, obs: np.ndarray) -> int: ...


class NoopPartner:
    """Always STAY. The 'no-opponent' baseline for world-model training."""
    name = "noop"
    def reset(self) -> None: pass
    def act(self, obs: np.ndarray) -> int: return 4


class RandomPartner:
    """Uniform random over 6 actions. Exposes dynamics to varied partner moves."""
    def __init__(self, seed: int | None = None, name: str = "random"):
        self.rng = np.random.default_rng(seed)
        self.name = name
    def reset(self) -> None: pass
    def act(self, obs: np.ndarray) -> int:
        return int(self.rng.integers(0, N_ACTIONS))


class DirectionPartner:
    """Persistent move in a single direction (with occasional STAY)."""
    def __init__(self, direction: int, stay_prob: float = 0.1,
                 seed: int | None = None, name: str | None = None):
        assert direction in (0, 1, 2, 3), "direction must be NORTH/SOUTH/EAST/WEST"
        self.direction = direction
        self.stay_prob = stay_prob
        self.rng = np.random.default_rng(seed)
        self.name = name or f"dir_{['N','S','E','W'][direction]}"
    def reset(self) -> None: pass
    def act(self, obs: np.ndarray) -> int:
        if self.rng.random() < self.stay_prob:
            return 4
        return self.direction


class WanderingPartner:
    """Random with momentum — same action for k steps, then re-roll."""
    def __init__(self, momentum: float = 0.7, interact_prob: float = 0.05,
                 seed: int | None = None, name: str = "wander"):
        self.momentum = momentum
        self.interact_prob = interact_prob
        self.rng = np.random.default_rng(seed)
        self._last: int | None = None
        self.name = name
    def reset(self) -> None:
        self._last = None
    def act(self, obs: np.ndarray) -> int:
        if self._last is None or self.rng.random() > self.momentum:
            if self.rng.random() < self.interact_prob:
                self._last = 5
            else:
                self._last = int(self.rng.integers(0, 5))  # avoid interact often
        return self._last


class MixedPartner:
    """Per-episode sample from a list of partners. The sampled partner is
    fixed for the duration of an episode (resets on ``reset``). Used to
    drive diverse trajectory collection for Phase-1 world-model training.
    """
    def __init__(self, partners: list[Partner], seed: int | None = None,
                 name: str = "mixed"):
        self.partners = partners
        self.rng = random.Random(seed)
        self._active: Partner = self.partners[0]
        self.name = name
    def reset(self) -> None:
        self._active = self.rng.choice(self.partners)
        self._active.reset()
    def act(self, obs: np.ndarray) -> int:
        return self._active.act(obs)
    @property
    def active_name(self) -> str:
        return self._active.name
