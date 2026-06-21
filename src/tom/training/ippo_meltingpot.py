"""Shared-parameter IPPO baseline for the MeltingPot Commons-Harvest substrate.

This is the standard MeltingPot self-play baseline: all N players share one
actor-critic (parameter sharing), trained with independent PPO on each player's
own egocentric stream. The PPO core — clipped policy loss, clipped value loss,
GAE, entropy bonus, orthogonal init — is the same recipe as
``ippo_overcooked`` / ``ippo_hanabi``; the only real differences are a small
**conv encoder** (observations are egocentric image stacks, not flat features)
and an **agent axis** folded into the batch so the shared net sees every
player's transition.

It is deliberately a clean baseline (no opponent-modeling aux head) — the OM /
SOM / TOM variants from ``ippo_overcooked`` are what you'd layer on top of this
to study partner modeling under a mixed-motive social dilemma.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from tom.envs.meltingpot_commons import VecMeltingPotCommonsEnv


# ---- Network -----------------------------------------------------------

def _orth(m: nn.Module, gain=np.sqrt(2)):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(m.weight, gain=gain)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class ConvActorCritic(nn.Module):
    """Conv encoder shared by a categorical policy head and a value head."""

    def __init__(self, obs_shape: tuple[int, int, int], n_actions: int, hidden: int = 256):
        super().__init__()
        c, h, w = obs_shape
        self.encoder = nn.Sequential(
            nn.Conv2d(c, 32, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(),
        )
        for m in self.encoder:
            _orth(m, gain=np.sqrt(2))
        with torch.no_grad():
            conv_out = self.encoder(torch.zeros(1, c, h, w)).shape[1]
        self.trunk = nn.Sequential(nn.Linear(conv_out, hidden), nn.ReLU())
        _orth(self.trunk[0], gain=np.sqrt(2))
        self.policy_head = nn.Linear(hidden, n_actions)
        _orth(self.policy_head, gain=0.01)
        self.value_head = nn.Linear(hidden, 1)
        _orth(self.value_head, gain=1.0)

    def _features(self, obs: torch.Tensor) -> torch.Tensor:
        return self.trunk(self.encoder(obs))

    def act(self, obs: torch.Tensor, deterministic: bool = False):
        h = self._features(obs)
        logits = self.policy_head(h)
        dist = torch.distributions.Categorical(logits=logits)
        a = logits.argmax(-1) if deterministic else dist.sample()
        return a, dist.log_prob(a), self.value_head(h).squeeze(-1)

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor):
        h = self._features(obs)
        dist = torch.distributions.Categorical(logits=self.policy_head(h))
        return dist.log_prob(actions), dist.entropy(), self.value_head(h).squeeze(-1)

    def value_only(self, obs: torch.Tensor) -> torch.Tensor:
        return self.value_head(self._features(obs)).squeeze(-1)


def compute_gae(rewards, values, dones, last_values, gamma=0.99, lam=0.95):
    T, N = rewards.shape
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros(N, device=rewards.device)
    for t in reversed(range(T)):
        next_v = last_values if t == T - 1 else values[t + 1]
        nonterm = 1.0 - dones[t].float()
        delta = rewards[t] + gamma * next_v * nonterm - values[t]
        gae = delta + gamma * lam * nonterm * gae
        advantages[t] = gae
    return advantages, advantages + values


# ---- Config ------------------------------------------------------------

@dataclass
class MeltingPotIPPOConfig:
    map_name: str = "default"
    num_players: int = 5
    num_envs: int = 16
    horizon: int = 1000
    view_radius: int = 5
    beam_length: int = 3
    freeze_steps: int = 25

    total_steps: int = 2_000_000
    rollout: int = 100
    learning_epochs: int = 4
    mini_batches: int = 4
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.1
    ent_coef: float = 0.02
    ent_coef_end: float | None = None
    ent_coef_horizon: int = 1_000_000
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden: int = 256

    log_dir: str = "runs_meltingpot/ippo"
    log_interval: int = 1
    ckpt_interval_steps: int = 500_000
    seed: int = 0
    device: str = "auto"
    resume_from: str | None = None


def _tensor(x, device):
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def train(cfg: MeltingPotIPPOConfig) -> str:
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if cfg.device == "auto"
        else torch.device(cfg.device)
    )
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    env = VecMeltingPotCommonsEnv(
        num_envs=cfg.num_envs,
        num_players=cfg.num_players,
        map_name=cfg.map_name,
        horizon=cfg.horizon,
        view_radius=cfg.view_radius,
        beam_length=cfg.beam_length,
        freeze_steps=cfg.freeze_steps,
        seed=cfg.seed,
    )
    P, C = cfg.num_players, env.obs_shape[0]
    B = cfg.num_envs * P  # shared net sees every player's transition

    net = ConvActorCritic(env.obs_shape, env.n_actions, cfg.hidden).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    n_params = sum(p.numel() for p in net.parameters())

    print(
        f"[cfg] substrate=commons_harvest/{cfg.map_name}  players={P}  "
        f"obs_shape={env.obs_shape}  n_actions={env.n_actions}  "
        f"num_envs={cfg.num_envs}  rollout={cfg.rollout}  B={B}  "
        f"params={n_params/1e3:.0f}K  total={cfg.total_steps}  device={device}"
    )

    if cfg.resume_from:
        ck = torch.load(cfg.resume_from, map_location=device)
        net.load_state_dict(ck["net"], strict=False)
        opt.load_state_dict(ck["opt"])
        start_step = int(ck.get("step", 0))
        print(f"[resume] loaded {cfg.resume_from} @ step {start_step}")
    else:
        start_step = 0

    os.makedirs(cfg.log_dir, exist_ok=True)
    writer = SummaryWriter(cfg.log_dir)

    def flat_obs(o):  # (num_envs, P, C, H, W) -> (B, C, H, W)
        return o.reshape(B, *env.obs_shape)

    obs = env.reset(seed=cfg.seed)  # (num_envs, P, C, H, W)
    obs_shape = env.obs_shape

    global_step = start_step
    iters = (cfg.total_steps - start_step) // (cfg.rollout * cfg.num_envs * P) + 1
    roll_coll, roll_equal, roll_pc, roll_apples = [], [], [], []
    t0 = time.time()

    for it in range(iters):
        if cfg.ent_coef_end is not None:
            frac = min(1.0, global_step / max(1, cfg.ent_coef_horizon))
            ent_coef = cfg.ent_coef + frac * (cfg.ent_coef_end - cfg.ent_coef)
        else:
            ent_coef = cfg.ent_coef

        T = cfg.rollout
        buf = {
            "obs": torch.zeros((T, B, *obs_shape), dtype=torch.float32, device=device),
            "actions": torch.zeros((T, B), dtype=torch.long, device=device),
            "logprobs": torch.zeros((T, B), dtype=torch.float32, device=device),
            "values": torch.zeros((T, B), dtype=torch.float32, device=device),
            "rewards": torch.zeros((T, B), dtype=torch.float32, device=device),
            "dones": torch.zeros((T, B), dtype=torch.float32, device=device),
        }

        for t in range(T):
            o = _tensor(flat_obs(obs), device)
            with torch.no_grad():
                act, logp, val = net.act(o)
            buf["obs"][t] = o
            buf["actions"][t] = act
            buf["logprobs"][t] = logp
            buf["values"][t] = val

            act_np = act.cpu().numpy().reshape(cfg.num_envs, P)
            obs, rew, _term, trunc, info = env.step(act_np)

            buf["rewards"][t] = _tensor(rew.reshape(B), device)
            buf["dones"][t] = _tensor(trunc.reshape(B).astype(np.float32), device)
            for c in info["completed"]:
                if c is not None:
                    roll_coll.append(c["collective_return"])
                    roll_equal.append(c["equality"])
                    roll_pc.append(c["per_capita_return"])
                    roll_apples.append(c["apples_remaining"])
            global_step += cfg.num_envs * P

        with torch.no_grad():
            last_values = net.value_only(_tensor(flat_obs(obs), device))
        advs, rets = compute_gae(
            buf["rewards"], buf["values"], buf["dones"], last_values,
            gamma=cfg.gamma, lam=cfg.gae_lambda,
        )

        obs_b = buf["obs"].reshape(T * B, *obs_shape)
        act_b = buf["actions"].reshape(T * B)
        logp_old = buf["logprobs"].reshape(T * B)
        val_b = buf["values"].reshape(T * B)
        adv_b = advs.reshape(T * B)
        ret_b = rets.reshape(T * B)
        adv_b = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)

        n = obs_b.shape[0]
        mb_size = n // cfg.mini_batches
        idx = np.arange(n)
        losses_p, losses_v, losses_e, kls = [], [], [], []
        for _ep in range(cfg.learning_epochs):
            np.random.shuffle(idx)
            for s in range(0, n, mb_size):
                mi = torch.as_tensor(idx[s : s + mb_size], device=device)
                logp_new, ent, v_new = net.evaluate(obs_b[mi], act_b[mi])
                ratio = torch.exp(logp_new - logp_old[mi])
                s1 = ratio * adv_b[mi]
                s2 = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv_b[mi]
                p_loss = -torch.min(s1, s2).mean()
                v_clip = val_b[mi] + torch.clamp(v_new - val_b[mi], -cfg.clip_eps, cfg.clip_eps)
                v_loss = 0.5 * torch.max((v_new - ret_b[mi]) ** 2, (v_clip - ret_b[mi]) ** 2).mean()
                ent_loss = -ent.mean()
                loss = p_loss + cfg.vf_coef * v_loss + ent_coef * ent_loss

                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
                opt.step()

                with torch.no_grad():
                    kls.append((logp_old[mi] - logp_new).mean().item())
                losses_p.append(p_loss.item())
                losses_v.append(v_loss.item())
                losses_e.append(ent_loss.item())

        elapsed = time.time() - t0
        fps = int((global_step - start_step) / max(elapsed, 1e-6))
        if it % cfg.log_interval == 0:
            writer.add_scalar("perf/fps", fps, global_step)
            writer.add_scalar("loss/pi", float(np.mean(losses_p)), global_step)
            writer.add_scalar("loss/v", float(np.mean(losses_v)), global_step)
            writer.add_scalar("loss/ent", float(np.mean(losses_e)), global_step)
            writer.add_scalar("kl", float(np.mean(kls)), global_step)
            writer.add_scalar("schedule/ent_coef", ent_coef, global_step)
            score = ""
            if roll_coll:
                mc = float(np.mean(roll_coll[-50:]))
                mpc = float(np.mean(roll_pc[-50:]))
                meq = float(np.mean(roll_equal[-50:]))
                mar = float(np.mean(roll_apples[-50:]))
                writer.add_scalar("ep/collective_return", mc, global_step)
                writer.add_scalar("ep/per_capita_return", mpc, global_step)
                writer.add_scalar("ep/equality", meq, global_step)
                writer.add_scalar("ep/apples_remaining", mar, global_step)
                score = f"coll={mc:7.1f}  per_capita={mpc:6.2f}  equality={meq:.2f}  apples_left={mar:4.1f}"
            print(f"[{global_step:>9d}/{cfg.total_steps}] fps={fps:>6d}  {score}")

        if (global_step // cfg.ckpt_interval_steps) > (
            (global_step - cfg.rollout * cfg.num_envs * P) // cfg.ckpt_interval_steps
        ):
            cpath = os.path.join(cfg.log_dir, f"ckpt_{global_step:09d}.pt")
            torch.save({"step": global_step, "net": net.state_dict(),
                        "opt": opt.state_dict(), "cfg": cfg.__dict__}, cpath)
            print(f"  → saved {cpath}")

        if global_step >= cfg.total_steps:
            break

    final = os.path.join(cfg.log_dir, "ckpt_final.pt")
    torch.save({"step": global_step, "net": net.state_dict(),
                "opt": opt.state_dict(), "cfg": cfg.__dict__}, final)
    writer.close()
    env.close()
    print(f"done. final ckpt: {final}")
    return final
