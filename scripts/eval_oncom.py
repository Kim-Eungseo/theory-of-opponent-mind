"""Evaluation protocols for an OnCOM-trained learner.

Three protocols:

    P1 (sample-eff)  — fix one held-out partner, run N episodes,
                       report sparse-return curve.
    P2 (sequential)  — encounter held-out partners in a fixed order,
                       M episodes each, then re-evaluate the first
                       partner to measure forgetting.
    P3 (latent)      — collect K-step trajectories from every
                       (train ∪ held-out) partner, embed via η, save
                       (z, name) pairs for downstream t-SNE / probe.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tom.envs.overcooked_solo import OvercookedSoloEnv  # noqa: E402
from tom.opponent_pool.pool import (  # noqa: E402
    add_checkpoint_partners,
    make_default_heldout_pool,
    make_default_train_pool,
)
from tom.training.oncom_trainer import OnComConfig, OnComLearner, _build_encoder  # noqa: E402


def _load_learner(ckpt_path: str, device: torch.device):
    ck = torch.load(ckpt_path, map_location=device)
    cfg_d = ck["cfg"]
    cfg = OnComConfig(**{k: cfg_d[k] for k in OnComConfig.__dataclass_fields__ if k in cfg_d})
    obs_dim = int(ck["obs_dim"])
    n_actions = int(ck["n_actions"])
    encoder, _ = _build_encoder(cfg, obs_dim, device)
    learner = OnComLearner(encoder, cfg, obs_dim, n_actions).to(device)
    learner.load_state_dict(ck["learner"], strict=False)
    learner.eval()
    return learner, cfg, obs_dim, n_actions


def _run_episode(learner, env, K, device):
    obs, _ = env.reset()
    hist_obs = np.zeros((1, K, env.obs_dim), dtype=np.float32)
    hist_act = np.zeros((1, K), dtype=np.int64)
    total_sparse = 0.0
    total_shaped = 0.0
    steps = 0
    while True:
        ob = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        ho = torch.as_tensor(hist_obs, dtype=torch.float32, device=device)
        ha = torch.as_tensor(hist_act, dtype=torch.long, device=device)
        with torch.no_grad():
            a, _, _, _ = learner.act(ob, ho, ha, deterministic=False)
        a = int(a.item())
        new_obs, r, term, trunc, info = env.step(a)
        total_sparse += info.get("sparse_r", 0.0) if info.get("sparse_r") is not None else 0.0
        total_shaped += info.get("shaped_r", 0.0) if info.get("shaped_r") is not None else 0.0
        # the solo env reward isn't broken into sparse/shaped by default; use raw r
        total_sparse += r  # fallback: solo env reward
        steps += 1
        # update history
        hist_obs = np.roll(hist_obs, -1, axis=1)
        hist_obs[0, -1, :] = info["partner_obs"]
        hist_act = np.roll(hist_act, -1, axis=1)
        hist_act[0, -1] = info["partner_action"]
        if term or trunc:
            break
        obs = new_obs
    return {"sparse": float(total_sparse), "shaped": float(total_shaped), "steps": steps}


def protocol_p1(learner, cfg, partner_factory, n_episodes: int, device):
    env = OvercookedSoloEnv(
        partner=partner_factory(),
        layout=cfg.layout, horizon=cfg.horizon, view_radius=cfg.view_radius,
        learner_idx=0,
    )
    returns = []
    for ep in range(n_episodes):
        r = _run_episode(learner, env, cfg.traj_K, device)
        returns.append(r["sparse"])
    env.close()
    return returns


def protocol_p3_latent(learner, cfg, pool, n_chunks: int, K: int, device):
    """Collect K-step trajectories from each partner and embed via η."""
    out: dict[str, list[list[float]]] = {}
    for idx in range(len(pool)):
        partner, name = pool.get(idx)
        env = OvercookedSoloEnv(
            partner=partner,
            layout=cfg.layout, horizon=cfg.horizon, view_radius=cfg.view_radius,
            learner_idx=0,
        )
        zs: list[list[float]] = []
        for _ in range(n_chunks):
            obs, _ = env.reset()
            hist_obs = np.zeros((1, K, env.obs_dim), dtype=np.float32)
            hist_act = np.zeros((1, K), dtype=np.int64)
            for _ in range(K):
                ob = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                ho = torch.as_tensor(hist_obs, dtype=torch.float32, device=device)
                ha = torch.as_tensor(hist_act, dtype=torch.long, device=device)
                with torch.no_grad():
                    a, _, _, _ = learner.act(ob, ho, ha)
                a = int(a.item())
                new_obs, _, term, trunc, info = env.step(a)
                hist_obs = np.roll(hist_obs, -1, axis=1)
                hist_obs[0, -1, :] = info["partner_obs"]
                hist_act = np.roll(hist_act, -1, axis=1)
                hist_act[0, -1] = info["partner_action"]
                obs = new_obs
                if term or trunc:
                    break
            with torch.no_grad():
                ho = torch.as_tensor(hist_obs, dtype=torch.float32, device=device)
                ha = torch.as_tensor(hist_act, dtype=torch.long, device=device)
                z = learner.traj(ho, ha).cpu().numpy().tolist()[0]
            zs.append(z)
        env.close()
        out[name] = zs
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="OnCOM checkpoint")
    ap.add_argument("--protocol", choices=["p1", "p3"], default="p1")
    ap.add_argument("--n-episodes", type=int, default=20)
    ap.add_argument("--n-chunks", type=int, default=8)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=None, help="JSON output file")
    ap.add_argument("--ckpt-pool", nargs="*", default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")
    learner, cfg, _, _ = _load_learner(args.ckpt, device)

    heldout = make_default_heldout_pool()
    train = make_default_train_pool()
    if args.ckpt_pool:
        add_checkpoint_partners(heldout, args.ckpt_pool, device="cpu")

    results: dict = {}
    if args.protocol == "p1":
        for i in range(len(heldout)):
            name = heldout.names[i]
            rets = protocol_p1(learner, cfg, lambda i=i: heldout.get(i)[0], args.n_episodes, device)
            results[name] = rets
            print(f"  [{name}] mean={np.mean(rets):.2f}  std={np.std(rets):.2f}")
    elif args.protocol == "p3":
        z_train = protocol_p3_latent(learner, cfg, train, args.n_chunks, cfg.traj_K, device)
        z_held  = protocol_p3_latent(learner, cfg, heldout, args.n_chunks, cfg.traj_K, device)
        results = {"train": z_train, "heldout": z_held}

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"saved {args.out}")
    else:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
