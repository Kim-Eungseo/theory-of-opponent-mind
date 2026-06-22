"""Wrapper around the **real** DeepMind ``dm-meltingpot`` substrates.

This is the genuine MeltingPot (dmlab2d-backed) substrate, not the pure-NumPy
re-implementation in :mod:`tom.envs.meltingpot_commons`. It requires the
``tom-meltingpot`` conda env (``dm-meltingpot`` + ``dmlab2d``, which only have
wheels for Python 3.10/3.11).

It exposes the *same array API* as :class:`VecMeltingPotCommonsEnv` so the IPPO
trainer in :mod:`tom.training.ippo_meltingpot` runs on either backend
unchanged — only the observation (real egocentric ``88×88×3`` RGB sprites) and
the network (a :class:`NatureActorCritic`) differ.

MeltingPot's native API is ``dm_env``: ``reset()``/``step(list_of_actions)``
return a ``TimeStep(step_type, reward, discount, observation)`` where
``observation`` is one dict per player (keys ``RGB``, ``READY_TO_SHOOT``,
``WORLD.RGB``, ``COLLECTIVE_REWARD``) and ``reward`` is a length-N array. We
keep the per-player ``RGB`` view, channels-first and scaled to ``[0, 1]``.
"""
from __future__ import annotations

from typing import Any

import numpy as np

DEFAULT_SUBSTRATE = "commons_harvest__open"


def _gini(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    if x.sum() <= 0:
        return 0.0
    x = np.sort(x)
    n = x.size
    idx = np.arange(1, n + 1)
    return float(np.sum((2 * idx - n - 1) * x) / (n * x.sum()))


class MeltingPotSubstrateEnv:
    """One real MeltingPot substrate behind an egocentric per-player API."""

    metadata = {"name": "meltingpot_substrate_v0"}

    def __init__(
        self,
        substrate_name: str = DEFAULT_SUBSTRATE,
        roles: tuple[str, ...] | None = None,
        seed: int | None = None,
    ):
        from meltingpot import substrate as mp_substrate  # lazy: heavy native dep

        self.substrate_name = substrate_name
        cfg = mp_substrate.get_config(substrate_name)
        self._roles = tuple(roles) if roles is not None else tuple(cfg.default_player_roles)
        self.num_players = len(self._roles)
        self._env = mp_substrate.build(substrate_name, roles=self._roles)

        aspec = self._env.action_spec()
        self.n_actions = int(aspec[0].num_values)
        ospec = self._env.observation_spec()
        h, w, c = ospec[0]["RGB"].shape  # (88, 88, 3)
        self.obs_shape = (int(c), int(h), int(w))  # channels-first
        self.obs_dim = int(c * h * w)

        self.possible_agents = [f"player_{i}" for i in range(self.num_players)]
        self.agents = list(self.possible_agents)
        self._rng = np.random.default_rng(seed)
        self._ep_return = np.zeros(self.num_players, dtype=np.float64)
        self._t = 0

    # ---- internal array API (used by the vec wrapper) ----
    def _obs(self, timestep) -> np.ndarray:
        out = np.empty((self.num_players, *self.obs_shape), dtype=np.float32)
        for i, o in enumerate(timestep.observation):
            rgb = np.asarray(o["RGB"], dtype=np.float32) / 255.0  # (H, W, C)
            out[i] = np.transpose(rgb, (2, 0, 1))  # (C, H, W)
        return out

    def _reset_arr(self, seed: int | None = None) -> np.ndarray:
        ts = self._env.reset()
        self._ep_return[:] = 0.0
        self._t = 0
        return self._obs(ts)

    def _step_arr(self, actions: np.ndarray):
        ts = self._env.step([int(a) for a in actions])
        rew = np.asarray(ts.reward, dtype=np.float32)
        self._ep_return += rew
        self._t += 1
        return self._obs(ts), rew, bool(ts.last())

    def _episode_stats(self) -> dict[str, float]:
        coll = float(self._ep_return.sum())
        return {
            "collective_return": coll,
            "per_capita_return": coll / self.num_players,
            "equality": 1.0 - _gini(self._ep_return),
            "length": int(self._t),
        }

    # ---- public dict API (parity with the other envs) ----
    def reset(self, seed: int | None = None, options: dict | None = None):
        obs = self._reset_arr(seed=seed)
        return (
            {a: obs[i] for i, a in enumerate(self.possible_agents)},
            {a: {} for a in self.possible_agents},
        )

    def step(self, actions: dict[str, int]):
        a = np.array([actions[a] for a in self.possible_agents], dtype=np.int64)
        obs, rew, truncated = self._step_arr(a)
        obs_d = {a: obs[i] for i, a in enumerate(self.possible_agents)}
        rew_d = {a: float(rew[i]) for i, a in enumerate(self.possible_agents)}
        term_d = {a: False for a in self.possible_agents}
        trunc_d = {a: bool(truncated) for a in self.possible_agents}
        stats = self._episode_stats() if truncated else {}
        info_d = {a: dict(stats) for a in self.possible_agents}
        return obs_d, rew_d, term_d, trunc_d, info_d

    def close(self) -> None:
        self._env.close()


class VecMeltingPotSubstrateEnv:
    """``num_envs`` real MeltingPot substrates stepped sequentially, auto-reset.

    Array API identical to :class:`VecMeltingPotCommonsEnv`: observations carry
    an explicit agent axis ``(num_envs, num_players, C, H, W)`` and episode
    summaries arrive in ``info["completed"]``.
    """

    def __init__(
        self,
        num_envs: int = 6,
        substrate_name: str = DEFAULT_SUBSTRATE,
        roles: tuple[str, ...] | None = None,
        seed: int = 0,
    ):
        self._num_envs = int(num_envs)
        self.envs = [
            MeltingPotSubstrateEnv(substrate_name=substrate_name, roles=roles, seed=seed + i)
            for i in range(num_envs)
        ]
        tpl = self.envs[0]
        self.substrate_name = substrate_name
        self.num_players = tpl.num_players
        self.possible_agents = tpl.possible_agents
        self.obs_shape = tpl.obs_shape
        self.obs_dim = tpl.obs_dim
        self.n_actions = tpl.n_actions
        self._last_obs = np.zeros((num_envs, tpl.num_players, *tpl.obs_shape), dtype=np.float32)

    @property
    def num_envs(self) -> int:
        return self._num_envs

    def reset(self, seed: int | None = None) -> np.ndarray:
        for i, e in enumerate(self.envs):
            self._last_obs[i] = e._reset_arr(seed=None if seed is None else seed + i)
        return self._last_obs.copy()

    def step(self, actions: np.ndarray):
        """actions: (num_envs, num_players) int array."""
        actions = np.asarray(actions, dtype=np.int64).reshape(self._num_envs, self.num_players)
        n, P = self._num_envs, self.num_players
        obs = np.zeros((n, P, *self.obs_shape), dtype=np.float32)
        rew = np.zeros((n, P), dtype=np.float32)
        term = np.zeros((n, P), dtype=bool)
        trunc = np.zeros((n, P), dtype=bool)
        completed: list[dict | None] = [None] * n
        for i in range(n):
            o_i, r_i, truncated = self.envs[i]._step_arr(actions[i])
            obs[i] = o_i
            rew[i] = r_i
            trunc[i] = truncated
            if truncated:
                completed[i] = self.envs[i]._episode_stats()
                obs[i] = self.envs[i]._reset_arr()  # auto-reset
        self._last_obs = obs.copy()
        return obs, rew, term, trunc, {"completed": completed}

    def close(self) -> None:
        for e in self.envs:
            e.close()
