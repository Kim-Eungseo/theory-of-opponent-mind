"""Minimal smoke test for the multi-agent env — 5 steps of random play."""
from __future__ import annotations

import socket

import numpy as np
import pytest


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.timeout(120)
def test_multi_agent_env_runs_a_few_steps():
    from tom.envs import VizDoomMultiAgentEnv

    env = VizDoomMultiAgentEnv(
        num_players=2,
        episode_timeout_seconds=5.0,
        port=_free_port(),
        seed=0,
    )
    try:
        obs, infos = env.reset()
        assert set(env.agents) == {"player_0", "player_1"}
        for a in env.agents:
            assert obs[a]["screen"].shape == (3, 84, 84)

        for _ in range(5):
            actions = {a: env.action_space(a).sample() for a in env.agents}
            obs, rewards, terms, truncs, infos = env.step(actions)
            for a in rewards:
                assert isinstance(rewards[a], float)
    finally:
        env.close()


@pytest.mark.timeout(60)
def test_meltingpot_commons_env_runs_a_full_episode():
    """Pure-NumPy substrate — no native deps, always runnable."""
    from tom.envs import MeltingPotCommonsEnv, VecMeltingPotCommonsEnv

    env = MeltingPotCommonsEnv(num_players=5, horizon=50, view_radius=5, seed=0)
    obs, infos = env.reset(seed=0)
    assert set(env.possible_agents) == {f"player_{i}" for i in range(5)}
    for a in env.possible_agents:
        assert obs[a].shape == env.obs_shape == (4, 11, 11)
        assert obs[a].dtype == np.float32

    rng = np.random.default_rng(0)
    truncated = False
    last_info = {}
    for _ in range(50):
        actions = {a: int(rng.integers(0, env.n_actions)) for a in env.possible_agents}
        obs, rewards, terms, truncs, infos = env.step(actions)
        for a in rewards:
            assert isinstance(rewards[a], float)
        truncated = any(truncs.values())
        last_info = infos
        if truncated:
            break
    assert truncated  # episode truncates at the horizon
    stats = last_info["player_0"]
    assert {"collective_return", "equality", "apples_remaining"} <= set(stats)
    env.close()

    # vectorised array API
    vec = VecMeltingPotCommonsEnv(num_envs=4, num_players=5, horizon=50, view_radius=5, seed=0)
    o = vec.reset(seed=0)
    assert o.shape == (4, 5, 4, 11, 11)
    for _ in range(50):
        a = rng.integers(0, vec.n_actions, size=(4, 5))
        o, r, term, trunc, info = vec.step(a)
        assert o.shape == (4, 5, 4, 11, 11)
        assert r.shape == (4, 5)
    assert any(c is not None for c in info["completed"])  # all 4 envs truncate together
    vec.close()
