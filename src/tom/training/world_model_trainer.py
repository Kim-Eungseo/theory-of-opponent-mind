"""Phase-1: train the world model in solo Overcooked.

Collect ``(o_t, a_own_t, a_opp_t, o_{t+1}, r_{t+1})`` tuples in parallel
solo envs (one partner sampled per env per episode from a *diverse
scripted pool*) and minimise

    L = MSE(ψ(φ(o_t), a_own, a_opp), sg(φ(o_{t+1})))
        + MSE(r̂(φ(o_t), a_own, a_opp), r_{t+1})

The learner's own actions are sampled uniformly at random during
collection — Phase-1 cares only about dynamics, not return-maximisation.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from tom.envs.overcooked_solo import VecOvercookedSoloEnv
from tom.opponent_pool.partners import (
    DirectionPartner,
    MixedPartner,
    NoopPartner,
    RandomPartner,
    WanderingPartner,
)
from tom.world_model.model import WorldModel


@dataclass
class WorldModelConfig:
    layout: str = "asymmetric_advantages"
    horizon: int = 400
    view_radius: int = 2
    num_envs: int = 32
    total_steps: int = 2_000_000
    rollout_steps: int = 400

    # model
    latent: int = 128
    hidden: int = 256
    decode: bool = False
    decoder_weight: float = 0.1

    # optim
    lr: float = 3e-4
    batch_size: int = 1024
    num_updates_per_rollout: int = 200
    buffer_size: int = 200_000
    max_grad_norm: float = 0.5

    log_dir: str = "runs_world_model/wm"
    log_interval: int = 1
    ckpt_interval_steps: int = 500_000
    seed: int = 0
    device: str = "auto"


def _build_mixed_partner_factory(seed_offset: int):
    """Returns a callable: int(env_idx) -> Partner producing a MixedPartner
    that samples from a 6-component scripted suite each episode."""
    def factory(env_idx: int):
        base = seed_offset + env_idx * 7919
        return MixedPartner(
            partners=[
                NoopPartner(),
                RandomPartner(seed=base + 1, name="random"),
                DirectionPartner(0, name="dir_N"),
                DirectionPartner(1, name="dir_S"),
                DirectionPartner(2, name="dir_E"),
                WanderingPartner(seed=base + 5, name="wander"),
            ],
            seed=base,
        )
    return factory


class ReplayBuffer:
    """Tiny ring buffer for (obs, a_own, a_opp, obs', r) tuples."""

    def __init__(self, capacity: int, obs_dim: int, device):
        self.capacity = int(capacity)
        self.device = device
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.a_own = np.zeros(capacity, dtype=np.int64)
        self.a_opp = np.zeros(capacity, dtype=np.int64)
        self.rew = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.full = False

    def __len__(self) -> int:
        return self.capacity if self.full else self.pos

    def add_batch(self, obs, a_own, a_opp, next_obs, rew):
        n = obs.shape[0]
        for i in range(n):
            p = self.pos
            self.obs[p] = obs[i]
            self.a_own[p] = a_own[i]
            self.a_opp[p] = a_opp[i]
            self.next_obs[p] = next_obs[i]
            self.rew[p] = rew[i]
            self.pos = (self.pos + 1) % self.capacity
            if self.pos == 0:
                self.full = True

    def sample(self, batch: int):
        n = len(self)
        idx = np.random.randint(0, n, size=batch)
        to = lambda x, dt=torch.float32: torch.as_tensor(x, dtype=dt, device=self.device)
        return (
            to(self.obs[idx]),
            to(self.a_own[idx], dt=torch.long),
            to(self.a_opp[idx], dt=torch.long),
            to(self.next_obs[idx]),
            to(self.rew[idx]),
        )


def train_world_model(cfg: WorldModelConfig) -> str:
    if cfg.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.device)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    env = VecOvercookedSoloEnv(
        num_envs=cfg.num_envs,
        partner_factory=_build_mixed_partner_factory(cfg.seed),
        layout=cfg.layout,
        horizon=cfg.horizon,
        view_radius=cfg.view_radius,
        learner_idx=0,
        seed=cfg.seed,
    )
    obs, _ = env.reset(seed=cfg.seed)
    obs_dim = env.obs_dim
    n_actions = env.n_actions

    wm = WorldModel(
        obs_dim=obs_dim, n_actions=n_actions,
        latent=cfg.latent, hidden=cfg.hidden,
        decode=cfg.decode, decoder_weight=cfg.decoder_weight,
    ).to(device)
    opt = torch.optim.Adam(wm.parameters(), lr=cfg.lr)
    buf = ReplayBuffer(cfg.buffer_size, obs_dim, device)

    os.makedirs(cfg.log_dir, exist_ok=True)
    writer = SummaryWriter(cfg.log_dir)
    print(f"[wm] obs_dim={obs_dim}  latent={cfg.latent}  decode={cfg.decode}")

    global_step = 0
    iterations = max(1, cfg.total_steps // (cfg.rollout_steps * cfg.num_envs) + 1)
    t0 = time.time()
    rng = np.random.default_rng(cfg.seed)

    for it in range(iterations):
        # ---- collect rollout with random learner policy ----
        T, N = cfg.rollout_steps, cfg.num_envs
        for _ in range(T):
            a_own = rng.integers(0, n_actions, size=N).astype(np.int64)
            new_obs, rew, term, trunc, info_list = env.step(a_own)
            a_opp = np.array([info_list[i]["partner_action"] for i in range(N)], dtype=np.int64)
            buf.add_batch(obs, a_own, a_opp, new_obs, rew)
            obs = new_obs
            global_step += N

        # ---- update world model ----
        losses_dyn, losses_rew = [], []
        for _ in range(cfg.num_updates_per_rollout):
            ob, ao, ap, nob, rw = buf.sample(cfg.batch_size)
            loss, logs = wm.loss(ob, ao, ap, nob, rw)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(wm.parameters(), cfg.max_grad_norm)
            opt.step()
            losses_dyn.append(logs["l_dyn"])
            losses_rew.append(logs["l_rew"])

        # ---- log ----
        elapsed = time.time() - t0
        fps = int(global_step / max(elapsed, 1e-6))
        ldyn = float(np.mean(losses_dyn))
        lrew = float(np.mean(losses_rew))
        writer.add_scalar("loss/dyn", ldyn, global_step)
        writer.add_scalar("loss/rew", lrew, global_step)
        writer.add_scalar("perf/fps", fps, global_step)
        writer.add_scalar("buffer/size", len(buf), global_step)
        print(f"[{global_step:>9d}/{cfg.total_steps}] fps={fps:>6d}  "
              f"l_dyn={ldyn:.4f}  l_rew={lrew:.4f}  buf={len(buf)}")

        # ---- ckpt ----
        if (global_step // cfg.ckpt_interval_steps) > (
            (global_step - T * N) // cfg.ckpt_interval_steps
        ):
            cpath = os.path.join(cfg.log_dir, f"wm_{global_step:09d}.pt")
            torch.save(
                {"step": global_step, "model": wm.state_dict(), "cfg": cfg.__dict__,
                 "obs_dim": obs_dim, "n_actions": n_actions},
                cpath,
            )
            print(f"  → saved {cpath}")

        if global_step >= cfg.total_steps:
            break

    final = os.path.join(cfg.log_dir, "wm_final.pt")
    torch.save(
        {"step": global_step, "model": wm.state_dict(), "cfg": cfg.__dict__,
         "obs_dim": obs_dim, "n_actions": n_actions},
        final,
    )
    writer.close()
    env.close()
    print(f"done. final ckpt: {final}")
    return final
