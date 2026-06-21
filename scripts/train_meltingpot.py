"""CLI for shared-parameter IPPO self-play on MeltingPot Commons-Harvest."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tom.training.ippo_meltingpot import MeltingPotIPPOConfig, train  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", type=str, default="default", help="substrate map name")
    ap.add_argument("--num-players", type=int, default=5)
    ap.add_argument("--num-envs", type=int, default=16)
    ap.add_argument("--horizon", type=int, default=1000)
    ap.add_argument("--view-radius", type=int, default=5,
                    help="egocentric window is (2R+1)x(2R+1)")
    ap.add_argument("--beam-length", type=int, default=3)
    ap.add_argument("--freeze-steps", type=int, default=25)

    ap.add_argument("--total-steps", type=int, default=2_000_000)
    ap.add_argument("--rollout", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--mini-batches", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--gae-lambda", type=float, default=0.95)
    ap.add_argument("--clip-eps", type=float, default=0.1)
    ap.add_argument("--ent-coef", type=float, default=0.02)
    ap.add_argument("--ent-coef-end", type=float, default=None,
                    help="if set, anneal entropy coef from --ent-coef → this over --ent-coef-horizon")
    ap.add_argument("--ent-coef-horizon", type=int, default=1_000_000)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--ckpt-interval", type=int, default=500_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-dir", type=str, default=None)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--resume-from", type=str, default=None)
    args = ap.parse_args()

    log_dir = args.log_dir or f"runs_meltingpot/ippo_commons_{args.map}_p{args.num_players}"

    cfg = MeltingPotIPPOConfig(
        map_name=args.map,
        num_players=args.num_players,
        num_envs=args.num_envs,
        horizon=args.horizon,
        view_radius=args.view_radius,
        beam_length=args.beam_length,
        freeze_steps=args.freeze_steps,
        total_steps=args.total_steps,
        rollout=args.rollout,
        learning_epochs=args.epochs,
        mini_batches=args.mini_batches,
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_eps=args.clip_eps,
        ent_coef=args.ent_coef,
        ent_coef_end=args.ent_coef_end,
        ent_coef_horizon=args.ent_coef_horizon,
        vf_coef=args.vf_coef,
        hidden=args.hidden,
        ckpt_interval_steps=args.ckpt_interval,
        seed=args.seed,
        log_dir=log_dir,
        device=args.device,
        resume_from=args.resume_from,
    )
    train(cfg)


if __name__ == "__main__":
    main()
