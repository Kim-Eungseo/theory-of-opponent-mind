"""CLI for shared-policy LSTM PPO self-play on 2-player Hanabi."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tom.training.ippo_hanabi_lstm import HanabiLSTMPPOConfig, train  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-name", type=str, default="Hanabi-Full-CardKnowledge")
    ap.add_argument("--total-steps", type=int, default=5_000_000)
    ap.add_argument("--num-envs", type=int, default=32)
    ap.add_argument("--rollout-steps", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--mini-batches", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--gae-lambda", type=float, default=0.95)
    ap.add_argument("--clip-eps", type=float, default=0.2)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--belief-coef", type=float, default=0.0,
                    help="Coefficient on belief auxiliary loss; 0.0 = vanilla LSTM baseline")
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--lstm-hidden", type=int, default=512)
    ap.add_argument("--ckpt-interval", type=int, default=500_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-dir", type=str, default=None)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--resume-from", type=str, default=None)
    args = ap.parse_args()

    log_dir = args.log_dir or f"runs_hanabi/ippo_lstm_{args.env_name.replace('-', '_').lower()}"

    cfg = HanabiLSTMPPOConfig(
        env_name=args.env_name,
        num_envs=args.num_envs,
        total_steps=args.total_steps,
        rollout_steps=args.rollout_steps,
        learning_epochs=args.epochs,
        mini_batches=args.mini_batches,
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_eps=args.clip_eps,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        belief_coef=args.belief_coef,
        hidden=args.hidden,
        lstm_hidden=args.lstm_hidden,
        ckpt_interval_steps=args.ckpt_interval,
        seed=args.seed,
        log_dir=log_dir,
        device=args.device,
        resume_from=args.resume_from,
    )
    train(cfg)


if __name__ == "__main__":
    main()
