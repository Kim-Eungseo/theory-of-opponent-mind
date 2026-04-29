"""Two-agent Overcooked-AI env wrapped in a PettingZoo-parallel-style API.

The underlying ``OvercookedEnv`` already exposes a 2-player joint action API,
so we just expose it through the dict-of-agent interface our trainer expects:
``reset()`` returns ``{agent: obs}`` and ``step({agent: action})`` returns
``(obs, reward, terminated, truncated, info)`` per-agent.

Reward: by default we expose ``sparse + 0.5 * shaped`` so the trainer sees a
dense learning signal. Both signals are kept in ``info`` so the eval script
can score on the sparse-only delivery reward (which is what Overcooked-AI
papers report).
"""
from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld

AGENT_IDS = ("agent_0", "agent_1")


class OvercookedMultiAgentEnv:
    """Cooperative 2-player Overcooked, dict API."""

    metadata = {"name": "overcooked_multi_v0"}

    def __init__(
        self,
        layout: str = "cramped_room",
        horizon: int = 400,
        shaped_reward_coef: float = 0.5,
        seed: int | None = None,
    ):
        self.layout = layout
        self.horizon = horizon
        self.shaped_reward_coef = float(shaped_reward_coef)
        self._rng = np.random.default_rng(seed)

        self.mdp = OvercookedGridworld.from_layout_name(layout)
        self.env = OvercookedEnv.from_mdp(self.mdp, horizon=horizon)

        self.possible_agents = list(AGENT_IDS)
        self.agents = list(AGENT_IDS)

        feat = self.env.featurize_state_mdp(self.env.state)
        self.obs_dim = int(feat[0].shape[0])
        self.n_actions = Action.NUM_ACTIONS  # 6

        self.observation_spaces = {
            a: gym.spaces.Box(-np.inf, np.inf, (self.obs_dim,), np.float32)
            for a in self.possible_agents
        }
        self.action_spaces = {
            a: gym.spaces.Discrete(self.n_actions) for a in self.possible_agents
        }

    # ---- API ----
    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.env.reset()
        obs = self._featurize()
        self.agents = list(self.possible_agents)
        return obs, {a: {} for a in self.agents}

    def step(
        self, actions: dict[str, int]
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, Any],
    ]:
        joint = tuple(Action.ALL_ACTIONS[int(actions[a])] for a in self.possible_agents)
        _state, sparse_r, done, info = self.env.step(joint)

        sparse_by_agent = info.get("sparse_r_by_agent", [sparse_r / 2.0, sparse_r / 2.0])
        shaped_by_agent = info.get("shaped_r_by_agent", [0.0, 0.0])

        rewards = {}
        for i, a in enumerate(self.possible_agents):
            rewards[a] = float(sparse_by_agent[i]) + self.shaped_reward_coef * float(
                shaped_by_agent[i]
            )

        obs = self._featurize()
        terms = {a: False for a in self.possible_agents}
        truncs = {a: bool(done) for a in self.possible_agents}
        infos = {
            a: {
                "sparse_r": float(sparse_by_agent[i]),
                "shaped_r": float(shaped_by_agent[i]),
                "ep_done": bool(done),
            }
            for i, a in enumerate(self.possible_agents)
        }
        return obs, rewards, terms, truncs, infos

    def _featurize(self) -> dict[str, np.ndarray]:
        feats = self.env.featurize_state_mdp(self.env.state)
        return {a: np.asarray(feats[i], dtype=np.float32) for i, a in enumerate(self.possible_agents)}

    def state(self) -> np.ndarray:
        feats = self.env.featurize_state_mdp(self.env.state)
        return np.concatenate([np.asarray(f, dtype=np.float32) for f in feats])

    def close(self):
        pass


class VecOvercookedEnv:
    """N independent Overcooked games stepped sequentially.

    Overcooked is so cheap that even a Python for-loop is faster than the
    threading/IPC overhead. Auto-resets sub-envs at episode end.
    """

    def __init__(
        self,
        num_envs: int = 16,
        layout: str = "cramped_room",
        horizon: int = 400,
        shaped_reward_coef: float = 0.5,
        seed: int = 0,
    ):
        self._num_envs = int(num_envs)
        self.envs = [
            OvercookedMultiAgentEnv(
                layout=layout,
                horizon=horizon,
                shaped_reward_coef=shaped_reward_coef,
                seed=seed + i,
            )
            for i in range(num_envs)
        ]
        tpl = self.envs[0]
        self.possible_agents = tpl.possible_agents
        self.observation_spaces = tpl.observation_spaces
        self.action_spaces = tpl.action_spaces
        self.obs_dim = tpl.obs_dim
        self.n_actions = tpl.n_actions
        self._sub_last_obs: list[dict[str, np.ndarray]] = [{} for _ in range(num_envs)]
        # episode bookkeeping
        self._ep_sparse_return = np.zeros(num_envs, dtype=np.float32)
        self._ep_shaped_return = np.zeros(num_envs, dtype=np.float32)
        self._ep_len = np.zeros(num_envs, dtype=np.int32)
        # last completed episode (filled on auto-reset)
        self._last_completed: list[dict | None] = [None] * num_envs

    @property
    def num_envs(self) -> int:
        return self._num_envs

    def reset(self, seed: int | None = None) -> dict[str, np.ndarray]:
        for i, e in enumerate(self.envs):
            obs, _ = e.reset(seed=None if seed is None else seed + i)
            self._sub_last_obs[i] = obs
        self._ep_sparse_return[:] = 0.0
        self._ep_shaped_return[:] = 0.0
        self._ep_len[:] = 0
        return self._stack_obs()

    def _stack_obs(self) -> dict[str, np.ndarray]:
        return {
            a: np.stack([self._sub_last_obs[i][a] for i in range(self._num_envs)])
            for a in self.possible_agents
        }

    def step(self, actions: dict[str, np.ndarray]):
        obs_arr = {a: np.zeros((self._num_envs, self.obs_dim), dtype=np.float32) for a in self.possible_agents}
        rew = {a: np.zeros(self._num_envs, dtype=np.float32) for a in self.possible_agents}
        term = {a: np.zeros(self._num_envs, dtype=bool) for a in self.possible_agents}
        trunc = {a: np.zeros(self._num_envs, dtype=bool) for a in self.possible_agents}
        completed: list[dict | None] = [None] * self._num_envs

        for i in range(self._num_envs):
            ai = {a: int(actions[a][i]) for a in self.possible_agents}
            o_i, r_i, t_i, tr_i, info_i = self.envs[i].step(ai)
            for a in self.possible_agents:
                obs_arr[a][i] = o_i[a]
                rew[a][i] = r_i[a]
                term[a][i] = t_i[a]
                trunc[a][i] = tr_i[a]
            self._ep_sparse_return[i] += sum(info_i[a]["sparse_r"] for a in self.possible_agents)
            self._ep_shaped_return[i] += sum(info_i[a]["shaped_r"] for a in self.possible_agents)
            self._ep_len[i] += 1

            done_i = any(t_i[a] or tr_i[a] for a in self.possible_agents)
            self._sub_last_obs[i] = o_i
            if done_i:
                completed[i] = {
                    "sparse_return": float(self._ep_sparse_return[i]),
                    "shaped_return": float(self._ep_shaped_return[i]),
                    "length": int(self._ep_len[i]),
                }
                self._ep_sparse_return[i] = 0.0
                self._ep_shaped_return[i] = 0.0
                self._ep_len[i] = 0
                rst, _ = self.envs[i].reset()
                self._sub_last_obs[i] = rst
                for a in self.possible_agents:
                    obs_arr[a][i] = rst[a]

        self._last_completed = completed
        infos = {"completed": completed}
        return obs_arr, rew, term, trunc, infos

    def close(self):
        for e in self.envs:
            e.close()
