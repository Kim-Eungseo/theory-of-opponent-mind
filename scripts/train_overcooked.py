"""CLI for IPPO self-play on multi-agent Overcooked."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tom.training.ippo_overcooked import IPPOConfig, train  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", type=str, default="cramped_room",
                    help="Overcooked layout (cramped_room / asymmetric_advantages / "
                         "coordination_ring / forced_coordination / counter_circuit_o_1order)")
    ap.add_argument("--total-steps", type=int, default=1_000_000)
    ap.add_argument("--num-envs", type=int, default=16)
    ap.add_argument("--rollout", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--mini-batches", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--gae-lambda", type=float, default=0.98)
    ap.add_argument("--clip-eps", type=float, default=0.05)
    ap.add_argument("--ent-coef", type=float, default=0.1)
    ap.add_argument("--ent-coef-end", type=float, default=None,
                    help="If set, anneal ent_coef from --ent-coef → --ent-coef-end over --ent-coef-horizon steps")
    ap.add_argument("--ent-coef-horizon", type=int, default=300_000)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--om-coef", type=float, default=0.0,
                    help="Partner-action prediction aux loss coefficient; 0 = vanilla baseline")
    ap.add_argument("--om-in-policy", action="store_true",
                    help="BAD/SAD-style: concat OM softmax into policy/value head input")
    ap.add_argument("--som-coef", type=float, default=0.0,
                    help="Self-Other Modeling: pass partner_obs through my own policy and "
                         "fit partner's action via NLL")
    ap.add_argument("--tom-coef", type=float, default=0.0,
                    help="Trajectory OM: separate LSTM head over partner's last K obs")
    ap.add_argument("--tom-history-len", type=int, default=8)
    ap.add_argument("--tom-hidden", type=int, default=128)
    ap.add_argument("--tom-in-policy", action="store_true",
                    help="Route trajectory-OM softmax into policy/value head input")
    ap.add_argument("--hidden", type=int, default=256,
                    help="Encoder hidden width (use larger to capacity-match TOM+BAD)")
    ap.add_argument("--shaped-anneal-frac", type=float, default=0.5)
    ap.add_argument("--ckpt-interval", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-dir", type=str, default=None)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--resume-from", type=str, default=None)
    args = ap.parse_args()

    log_dir = args.log_dir or f"runs_overcooked/ippo_{args.layout}"

    cfg = IPPOConfig(
        layout=args.layout,
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
        om_coef=args.om_coef,
        om_in_policy=args.om_in_policy,
        som_coef=args.som_coef,
        tom_coef=args.tom_coef,
        tom_history_len=args.tom_history_len,
        tom_hidden=args.tom_hidden,
        tom_in_policy=args.tom_in_policy,
        hidden=args.hidden,
        shaped_reward_anneal_frac=args.shaped_anneal_frac,
        ckpt_interval_steps=args.ckpt_interval,
        seed=args.seed,
        log_dir=log_dir,
        device=args.device,
        resume_from=args.resume_from,
    )
    train(cfg)


if __name__ == "__main__":
    main()
