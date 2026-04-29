"""CLI entry point for skrl IPPO self-play training."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tom.training.skrl_ppo import SkrlPPOConfig, train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--total-steps", type=int, default=50_000)
    ap.add_argument("--rollout", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--mini-batches", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2.5e-4)
    ap.add_argument("--episode-seconds", type=float, default=720.0)
    ap.add_argument("--ticrate", type=int, default=1000)
    ap.add_argument("--scenario", type=str, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-dir", type=str, default="runs/skrl")
    ap.add_argument("--ckpt-interval", type=int, default=10000)
    ap.add_argument("--num-envs", type=int, default=4,
                    help="Parallel env instances; each holds 2 vizdoom workers")
    ap.add_argument("--resume-from", type=str, default=None,
                    help="Path to skrl IPPO checkpoint to continue training from")
    ap.add_argument("--mixed-precision", action="store_true",
                    help="Enable AMP (autocast) during PPO updates")
    args = ap.parse_args()

    scenario = args.scenario
    if scenario and not scenario.endswith(".cfg") and not scenario.startswith("/"):
        scenario = scenario + ".cfg"
    if scenario and not scenario.startswith("/"):
        import vizdoom as vzd
        scenario = str(Path(vzd.scenarios_path) / scenario)

    cfg = SkrlPPOConfig(
        total_steps=args.total_steps,
        rollout=args.rollout,
        learning_epochs=args.epochs,
        mini_batches=args.mini_batches,
        lr=args.lr,
        log_dir=args.log_dir,
        ckpt_interval=args.ckpt_interval,
        num_envs=args.num_envs,
        resume_from=args.resume_from,
        mixed_precision=args.mixed_precision,
    )
    train(
        cfg,
        env_kwargs=dict(
            episode_timeout_seconds=args.episode_seconds,
            ticrate=args.ticrate,
            scenario=scenario,
        ),
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
