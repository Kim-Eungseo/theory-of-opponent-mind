"""Launch self-play PPO training on VizDoomMultiAgentEnv.

Short runs to verify learning; long runs for real training. Tune via CLI.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tom.training.ppo import PPOConfig, train


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--total-steps", type=int, default=20_000)
    ap.add_argument("--rollout-steps", type=int, default=128)
    ap.add_argument("--episode-seconds", type=float, default=20.0)
    ap.add_argument("--ticrate", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=2.5e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", type=str, default=None,
                    help="path to save policy .pt (final + periodic)")
    args = ap.parse_args()

    cfg = PPOConfig(
        total_steps=args.total_steps,
        rollout_steps=args.rollout_steps,
        lr=args.lr,
        save_path=args.save,
    )
    train(
        cfg,
        env_kwargs=dict(
            episode_timeout_seconds=args.episode_seconds,
            ticrate=args.ticrate,
        ),
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
