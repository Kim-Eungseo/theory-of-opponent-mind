"""Minimal smoke test for the multi-agent env — 5 steps of random play."""
from __future__ import annotations

import socket

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
