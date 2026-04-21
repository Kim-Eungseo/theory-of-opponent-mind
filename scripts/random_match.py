"""Run one random-vs-random match on the multi-agent ViZDoom env."""
from __future__ import annotations

import argparse
import time

from tom.agents import RandomAgent
from tom.envs import VizDoomMultiAgentEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--players", type=int, default=2)
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--port", type=int, default=5029)
    args = parser.parse_args()

    env = VizDoomMultiAgentEnv(
        num_players=args.players,
        episode_timeout_seconds=args.seconds,
        port=args.port,
        seed=args.seed,
    )
    obs, infos = env.reset()
    agents = {a: RandomAgent(env.action_space(a), seed=args.seed + i)
              for i, a in enumerate(env.agents)}

    print(f"match start: agents={env.agents}")
    t0 = time.time()
    step = 0
    ep_return = {a: 0.0 for a in env.agents}

    while env.agents:
        actions = {a: agents[a].act(obs[a]) for a in env.agents}
        obs, rewards, terms, truncs, infos = env.step(actions)
        for a, r in rewards.items():
            ep_return[a] += r
        step += 1
        if step % 25 == 0:
            frags = {a: infos[a]["frags"] for a in infos}
            print(f"  step={step:4d} frags={frags} returns={ {a: round(v,2) for a,v in ep_return.items()} }")

    env.close()
    dt = time.time() - t0
    print(f"match done in {dt:.1f}s — {step} steps")
    print(f"returns: {ep_return}")
    print(f"final frags: { {a: infos[a]['frags'] for a in infos} }")


if __name__ == "__main__":
    main()
