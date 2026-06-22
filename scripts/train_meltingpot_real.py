"""CLI: shared-parameter IPPO on the REAL dm-meltingpot substrate.

Run inside the ``tom-meltingpot`` conda env (dm-meltingpot + dmlab2d + torch):

    conda run -n tom-meltingpot env PYTHONPATH=src \
        python scripts/train_meltingpot_real.py --substrate commons_harvest__open

Reuses the exact PPO core from ``ippo_meltingpot`` (the same trainer as the
NumPy substrate) — only the env (real dmlab2d) and the conv backbone
(NatureActorCritic, for 88×88 RGB) are swapped in.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tom.envs.meltingpot_substrate import VecMeltingPotSubstrateEnv  # noqa: E402
from tom.training.ippo_meltingpot import (  # noqa: E402
    MeltingPotIPPOConfig,
    NatureActorCritic,
    train,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--substrate", type=str, default="commons_harvest__open")
    ap.add_argument("--num-envs", type=int, default=6)
    ap.add_argument("--total-steps", type=int, default=400_000)
    ap.add_argument("--rollout", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--mini-batches", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--gae-lambda", type=float, default=0.95)
    ap.add_argument("--clip-eps", type=float, default=0.1)
    ap.add_argument("--ent-coef", type=float, default=0.02)
    ap.add_argument("--ent-coef-end", type=float, default=None)
    ap.add_argument("--ent-coef-horizon", type=int, default=400_000)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--ckpt-interval", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-dir", type=str, default=None)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--resume-from", type=str, default=None)
    args = ap.parse_args()

    log_dir = args.log_dir or f"runs_meltingpot/real_ippo_{args.substrate}"

    env = VecMeltingPotSubstrateEnv(
        num_envs=args.num_envs,
        substrate_name=args.substrate,
        seed=args.seed,
    )
    net = NatureActorCritic(env.obs_shape, env.n_actions, args.hidden)

    cfg = MeltingPotIPPOConfig(
        num_envs=args.num_envs,
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
    train(cfg, env=env, net=net)


if __name__ == "__main__":
    main()
