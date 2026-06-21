"""CLI for Phase-1 world-model training in solo Overcooked."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tom.training.world_model_trainer import WorldModelConfig, train_world_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", default="asymmetric_advantages")
    ap.add_argument("--horizon", type=int, default=400)
    ap.add_argument("--view-radius", type=int, default=2)
    ap.add_argument("--num-envs", type=int, default=32)
    ap.add_argument("--total-steps", type=int, default=2_000_000)
    ap.add_argument("--rollout-steps", type=int, default=400)
    ap.add_argument("--latent", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--decode", action="store_true",
                    help="add a reconstruction decoder loss")
    ap.add_argument("--decoder-weight", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--num-updates-per-rollout", type=int, default=200)
    ap.add_argument("--buffer-size", type=int, default=200_000)
    ap.add_argument("--ckpt-interval", type=int, default=500_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-dir", default=None)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    log_dir = args.log_dir or f"runs_world_model/wm_{args.layout}_K{args.view_radius}"
    cfg = WorldModelConfig(
        layout=args.layout,
        horizon=args.horizon,
        view_radius=args.view_radius,
        num_envs=args.num_envs,
        total_steps=args.total_steps,
        rollout_steps=args.rollout_steps,
        latent=args.latent,
        hidden=args.hidden,
        decode=args.decode,
        decoder_weight=args.decoder_weight,
        lr=args.lr,
        batch_size=args.batch_size,
        num_updates_per_rollout=args.num_updates_per_rollout,
        buffer_size=args.buffer_size,
        ckpt_interval_steps=args.ckpt_interval,
        seed=args.seed,
        log_dir=log_dir,
        device=args.device,
    )
    train_world_model(cfg)


if __name__ == "__main__":
    main()
