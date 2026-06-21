"""Single-agent wrapper around the 2-player Overcooked env.

Hides the partner behind a fixed ``Partner`` policy. The learner sees only
their own observation, picks their own action; the partner acts internally
and the joint step is executed. Tracks the partner's action so it can be
logged into the world-model replay buffer (Phase 1) or fed to the
opponent trajectory encoder (Phase 2).

Partial observability (``view_radius``) is inherited from the underlying
multi-agent env and applied per-agent.
"""
from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from tom.envs.overcooked_multi import OvercookedMultiAgentEnv, AGENT_IDS
from tom.opponent_pool.partners import Partner, NoopPartner


class OvercookedSoloEnv:
    """Single-agent env. The learner is ``learner_idx`` (0 or 1)."""

    metadata = {"name": "overcooked_solo_v0"}

    def __init__(
        self,
        partner: Partner | None = None,
        learner_idx: int = 0,
        layout: str = "asymmetric_advantages",
        horizon: int = 400,
        shaped_reward_coef: float = 0.5,
        view_radius: int | None = 2,
        seed: int | None = None,
    ):
        assert learner_idx in (0, 1)
        self.learner_idx = learner_idx
        self.partner_idx = 1 - learner_idx
        self.learner_id = AGENT_IDS[learner_idx]
        self.partner_id = AGENT_IDS[self.partner_idx]
        self.partner: Partner = partner if partner is not None else NoopPartner()

        self.base = OvercookedMultiAgentEnv(
            layout=layout,
            horizon=horizon,
            shaped_reward_coef=shaped_reward_coef,
            view_radius=view_radius,
            seed=seed,
        )

        self.obs_dim = self.base.obs_dim
        self.n_actions = self.base.n_actions
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, (self.obs_dim,), np.float32
        )
        self.action_space = gym.spaces.Discrete(self.n_actions)
        self._last_obs: dict[str, np.ndarray] = {}

    def set_partner(self, partner: Partner) -> None:
        self.partner = partner

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        obs, _ = self.base.reset(seed=seed)
        self.partner.reset()
        self._last_obs = obs
        return obs[self.learner_id], {"partner_obs": obs[self.partner_id]}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        partner_obs = self._last_obs[self.partner_id]
        a_opp = int(self.partner.act(partner_obs))
        joint = {self.learner_id: int(action), self.partner_id: a_opp}
        obs, rew, term, trunc, info = self.base.step(joint)
        self._last_obs = obs
        # bundle per-agent rewards / completion into a flat info dict
        learner_info: dict[str, Any] = {
            "partner_action": a_opp,
            "partner_obs": obs[self.partner_id],
            "partner_reward": float(rew[self.partner_id]),
            "ep_done": bool(term[self.learner_id] or trunc[self.learner_id]),
        }
        return (
            obs[self.learner_id],
            float(rew[self.learner_id]),
            bool(term[self.learner_id]),
            bool(trunc[self.learner_id]),
            learner_info,
        )

    def close(self) -> None:
        self.base.close()


class VecOvercookedSoloEnv:
    """N parallel solo envs. Each may have a different partner."""

    def __init__(
        self,
        num_envs: int,
        partner_factory,  # callable(env_idx) -> Partner
        layout: str = "asymmetric_advantages",
        horizon: int = 400,
        shaped_reward_coef: float = 0.5,
        view_radius: int | None = 2,
        learner_idx: int = 0,
        seed: int = 0,
    ):
        self._num_envs = int(num_envs)
        self.envs = [
            OvercookedSoloEnv(
                partner=partner_factory(i),
                learner_idx=learner_idx,
                layout=layout,
                horizon=horizon,
                shaped_reward_coef=shaped_reward_coef,
                view_radius=view_radius,
                seed=seed + i,
            )
            for i in range(num_envs)
        ]
        self.obs_dim = self.envs[0].obs_dim
        self.n_actions = self.envs[0].n_actions

    @property
    def num_envs(self) -> int:
        return self._num_envs

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, list[dict]]:
        obs_list, info_list = [], []
        for i, e in enumerate(self.envs):
            s = None if seed is None else seed + i
            o, info = e.reset(seed=s)
            obs_list.append(o)
            info_list.append(info)
        return np.stack(obs_list), info_list

    def step(self, actions: np.ndarray):
        obs_list, rew_list, term_list, trunc_list, info_list = [], [], [], [], []
        for i, e in enumerate(self.envs):
            o, r, term, trunc, info = e.step(int(actions[i]))
            # auto-reset on episode end so subsequent steps don't hit a done env
            if term or trunc:
                o, _ = e.reset()
            obs_list.append(o)
            rew_list.append(r)
            term_list.append(term)
            trunc_list.append(trunc)
            info_list.append(info)
        return (
            np.stack(obs_list),
            np.array(rew_list, dtype=np.float32),
            np.array(term_list, dtype=bool),
            np.array(trunc_list, dtype=bool),
            info_list,
        )

    def set_partner_for(self, env_i: int, partner: Partner) -> None:
        self.envs[env_i].set_partner(partner)

    def close(self) -> None:
        for e in self.envs:
            e.close()
