"""Vectorised wrapper around ``VizDoomMultiAgentEnv``.

Runs ``num_envs`` independent matches in parallel. Each sub-env keeps its own
two ViZDoom worker processes, so with num_envs=N we have 2N game processes
plus the parent. Sub-envs are stepped in parallel via Python threads — the
work is IO-bound (pipe recv), so the GIL is released and the threads overlap
cleanly.

Auto-resets a sub-env the moment its episode ends so the skrl vectorised
trainer never has to intervene.

Exposes the same attributes the skrl PettingZoo wrapper needs, but reports
``num_envs > 1`` and returns per-agent observations / rewards / flags already
stacked along the env dimension.
"""
from __future__ import annotations

import threading
from typing import Any

import gymnasium as gym
import numpy as np

from tom.envs.vizdoom_multi import VizDoomMultiAgentEnv


def _run_in_parallel(targets: list):
    threads = []
    for i, fn in enumerate(targets):
        t = threading.Thread(target=fn, name=f"vec-{i}")
        t.start()
        threads.append(t)
    for t in threads:
        t.join()


class VecVizDoomMultiAgentEnv:
    """Vectorised PettingZoo-parallel-style multi-agent Doom env."""

    metadata = {"name": "vec_vizdoom_multi_v0"}

    def __init__(
        self,
        num_envs: int = 4,
        num_players: int = 2,
        scenario: str | None = None,
        episode_timeout_seconds: float = 720.0,
        frame_skip: int = 4,
        frame_shape: tuple[int, int] = (84, 84),
        ticrate: int = 1000,
        seed: int | None = None,
        bots_filled: int = 0,
    ):
        if num_envs < 1:
            raise ValueError("num_envs must be >= 1")
        self._num_envs = int(num_envs)
        self.num_players = num_players
        self.frame_shape = tuple(frame_shape)

        base_seed = seed if seed is not None else 0
        # save kwargs so a crashed sub-env can be rebuilt on the fly
        self._base_kwargs = dict(
            num_players=num_players,
            scenario=scenario,
            episode_timeout_seconds=episode_timeout_seconds,
            frame_skip=frame_skip,
            frame_shape=frame_shape,
            ticrate=ticrate,
            bots_filled=bots_filled,
        )
        self._base_seed = base_seed
        self.envs: list[VizDoomMultiAgentEnv] = [
            self._make_sub_env(i) for i in range(num_envs)
        ]

        tpl = self.envs[0]
        self.possible_agents: list[str] = list(tpl.possible_agents)
        self.agents: list[str] = list(self.possible_agents)
        self.observation_spaces = tpl.observation_spaces
        self.action_spaces = tpl.action_spaces
        self.n_buttons = tpl.n_buttons
        self.n_vars = tpl.n_vars

        self._sub_last_obs: list[dict[str, dict[str, np.ndarray]]] = [{} for _ in range(num_envs)]

    def _make_sub_env(self, i: int) -> VizDoomMultiAgentEnv:
        return VizDoomMultiAgentEnv(
            port=None,
            seed=self._base_seed + i * 10007,
            **self._base_kwargs,
        )

    def _rebuild_sub_env(self, i: int) -> None:
        try:
            self.envs[i].close()
        except Exception:
            pass
        self.envs[i] = self._make_sub_env(i)
        obs, _ = self.envs[i].reset()
        self._sub_last_obs[i] = obs

    # ---- skrl-compatibility shims ----
    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def num_agents(self) -> int:
        return len(self.possible_agents)

    @property
    def max_num_agents(self) -> int:
        return len(self.possible_agents)

    @property
    def state_space(self) -> gym.spaces.Space:
        return self.envs[0].state_space

    def observation_space(self, agent: str):
        return self.observation_spaces[agent]

    def action_space(self, agent: str):
        return self.action_spaces[agent]

    # ---- core API ----
    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict]]:
        results: list[tuple[dict, dict] | None] = [None] * self._num_envs

        def _reset_one(i: int):
            def _call():
                s = None if seed is None else seed + i * 10007
                results[i] = self.envs[i].reset(seed=s)
            return _call

        _run_in_parallel([_reset_one(i) for i in range(self._num_envs)])
        for i, r in enumerate(results):
            if r is not None:
                self._sub_last_obs[i] = r[0]

        self.agents = list(self.possible_agents)
        return self._stack_obs(), {a: {} for a in self.agents}

    def _stack_obs(self) -> dict[str, dict[str, np.ndarray]]:
        out: dict[str, dict[str, np.ndarray]] = {}
        for agent in self.possible_agents:
            screens = np.stack([self._sub_last_obs[i][agent]["screen"] for i in range(self._num_envs)])
            gvs = np.stack([self._sub_last_obs[i][agent]["gamevars"] for i in range(self._num_envs)])
            out[agent] = {"screen": screens, "gamevars": gvs}
        return out

    def step(self, actions: dict[str, Any]):
        # actions[agent] is array-like of length num_envs (ints for Discrete)
        per_env_actions: list[dict[str, int]] = []
        for i in range(self._num_envs):
            ae = {}
            for agent in self.possible_agents:
                raw = actions[agent]
                if hasattr(raw, "__len__"):
                    ae[agent] = int(np.asarray(raw).reshape(-1)[i])
                else:
                    ae[agent] = int(raw)
            per_env_actions.append(ae)

        step_results: list[tuple | None] = [None] * self._num_envs

        def _step_one(i: int):
            def _call():
                try:
                    step_results[i] = self.envs[i].step(per_env_actions[i])
                except Exception as exc:  # noqa: BLE001 — re-raise after join
                    step_results[i] = ("ERROR", exc)
            return _call

        _run_in_parallel([_step_one(i) for i in range(self._num_envs)])

        # rebuild any sub-env that threw (usually a ViZDoom network timeout).
        # We synthesize a terminal step for the agents so skrl sees a clean
        # end-of-episode and the rollout continues.
        for i, r in enumerate(step_results):
            if r is None or (isinstance(r, tuple) and r and r[0] == "ERROR"):
                err = r[1] if (isinstance(r, tuple) and len(r) > 1) else RuntimeError("sub-env step returned None")
                print(
                    f"[vec] sub-env {i} crashed ({type(err).__name__}: {err}); rebuilding",
                    flush=True,
                )
                self._rebuild_sub_env(i)
                term_all = {a: True for a in self.possible_agents}
                trunc_all = {a: False for a in self.possible_agents}
                zero_reward = {a: 0.0 for a in self.possible_agents}
                info_all = {a: {"recovered": True} for a in self.possible_agents}
                step_results[i] = (self._sub_last_obs[i], zero_reward, term_all, trunc_all, info_all)

        # auto-reset terminated sub-envs (non-crash path)
        reset_indices = []
        for i, r in enumerate(step_results):
            obs_i, _rew, term_i, trunc_i, _info = r
            self._sub_last_obs[i] = obs_i
            if all(term_i.get(a, False) or trunc_i.get(a, False) for a in self.possible_agents):
                # the rebuild path already reset; skip to avoid double-reset
                if _info.get(next(iter(self.possible_agents)), {}).get("recovered"):
                    continue
                reset_indices.append(i)

        if reset_indices:
            def _reset_one(i: int):
                def _call():
                    rst = self.envs[i].reset()
                    self._sub_last_obs[i] = rst[0]
                return _call
            _run_in_parallel([_reset_one(i) for i in reset_indices])

        # stitch batched tensors
        obs = self._stack_obs()
        rewards: dict[str, np.ndarray] = {a: np.zeros(self._num_envs, dtype=np.float32) for a in self.possible_agents}
        terms: dict[str, np.ndarray] = {a: np.zeros(self._num_envs, dtype=bool) for a in self.possible_agents}
        truncs: dict[str, np.ndarray] = {a: np.zeros(self._num_envs, dtype=bool) for a in self.possible_agents}
        infos: dict[str, Any] = {}

        for i, r in enumerate(step_results):
            _obs, rew, term, trunc, info = r
            for agent in self.possible_agents:
                rewards[agent][i] = float(rew.get(agent, 0.0))
                terms[agent][i] = bool(term.get(agent, False))
                truncs[agent][i] = bool(trunc.get(agent, False))

        self.agents = list(self.possible_agents)
        return obs, rewards, terms, truncs, infos

    def state(self) -> dict[str, np.ndarray]:
        # Dict{screen, gamevars} with a leading num_envs dimension.
        states = [env.state() for env in self.envs]
        return {
            "screen": np.stack([s["screen"] for s in states]),
            "gamevars": np.stack([s["gamevars"] for s in states]),
        }

    def render(self) -> Any:
        return self.envs[0].render()

    def close(self) -> None:
        for env in self.envs:
            try:
                env.close()
            except Exception:
                pass
