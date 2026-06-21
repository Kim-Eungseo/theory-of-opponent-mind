"""CLI for Phase-2 OnCOM training in (partial-obs) Overcooked.

Uses the default scripted opponent pool (NOOP + Direction × 4 + Wander × 3
+ Random × 2) unless ``--ckpt-pool`` is given with a list of additional
checkpoint partner paths.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tom.opponent_pool.pool import (  # noqa: E402
    add_checkpoint_partners,
    make_default_heldout_pool,
    make_default_train_pool,
)
from tom.training.oncom_trainer import OnComConfig, train_oncom  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", default="asymmetric_advantages")
    ap.add_argument("--horizon", type=int, default=400)
    ap.add_argument("--view-radius", type=int, default=2)
    ap.add_argument("--num-envs", type=int, default=16)
    ap.add_argument("--total-steps", type=int, default=5_000_000)
    ap.add_argument("--rollout-steps", type=int, default=400)

    # encoder transfer
    ap.add_argument("--wm-ckpt", default=None,
                    help="Phase-1 world-model checkpoint to initialize φ")
    ap.add_argument("--encoder-mode", choices=["frozen", "lora", "full"], default="lora")
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=float, default=16.0)
    ap.add_argument("--latent-dim", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=256)

    # trajectory encoder
    ap.add_argument("--traj-K", type=int, default=16)
    ap.add_argument("--z-dim", type=int, default=32)
    ap.add_argument("--traj-hidden", type=int, default=128)

    # PPO
    ap.add_argument("--lr-policy", type=float, default=3e-4)
    ap.add_argument("--lr-encoder", type=float, default=1e-4)
    ap.add_argument("--lr-traj", type=float, default=3e-4)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--mini-batches", type=int, default=4)
    ap.add_argument("--clip-eps", type=float, default=0.05)
    ap.add_argument("--ent-coef", type=float, default=0.1)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--gae-lambda", type=float, default=0.98)

    # aux losses
    ap.add_argument("--om-coef", type=float, default=0.5)
    ap.add_argument("--contrastive-coef", type=float, default=0.5)
    ap.add_argument("--contrastive-temp", type=float, default=0.1)
    ap.add_argument("--momentum", type=float, default=0.99)

    # ckpt partner pool
    ap.add_argument("--ckpt-pool", nargs="*", default=None,
                    help="paths to ippo_overcooked checkpoints to add as partners")
    ap.add_argument("--partner-slot", default="agent_1",
                    help="slot key inside ippo checkpoint to load as partner")

    ap.add_argument("--ckpt-interval", type=int, default=500_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-dir", default=None)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    log_dir = args.log_dir or f"runs_oncom/oncom_{args.layout}_K{args.view_radius}_{args.encoder_mode}"
    cfg = OnComConfig(
        layout=args.layout,
        horizon=args.horizon,
        view_radius=args.view_radius,
        num_envs=args.num_envs,
        total_steps=args.total_steps,
        rollout_steps=args.rollout_steps,
        wm_ckpt=args.wm_ckpt,
        encoder_mode=args.encoder_mode,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        latent_dim=args.latent_dim,
        hidden=args.hidden,
        traj_K=args.traj_K,
        z_dim=args.z_dim,
        traj_hidden=args.traj_hidden,
        lr_policy=args.lr_policy,
        lr_encoder=args.lr_encoder,
        lr_traj=args.lr_traj,
        learning_epochs=args.epochs,
        mini_batches=args.mini_batches,
        clip_eps=args.clip_eps,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        om_coef=args.om_coef,
        contrastive_coef=args.contrastive_coef,
        contrastive_temperature=args.contrastive_temp,
        momentum=args.momentum,
        ckpt_interval_steps=args.ckpt_interval,
        seed=args.seed,
        log_dir=log_dir,
        device=args.device,
    )

    train_pool = make_default_train_pool(seed_base=args.seed)
    if args.ckpt_pool:
        add_checkpoint_partners(
            train_pool, args.ckpt_pool,
            partner_slot=args.partner_slot,
            device="cpu",
        )

    print(f"[oncom-cli] train pool ({len(train_pool)}): {train_pool.names}")
    train_oncom(cfg, train_pool)


if __name__ == "__main__":
    main()
