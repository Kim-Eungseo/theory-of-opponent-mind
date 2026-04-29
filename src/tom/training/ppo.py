"""Single-file self-play PPO for the multi-agent ViZDoom env.

Both players share one policy — classic self-play. Each env.step() yields two
transitions (one per agent), which we treat as if they came from two parallel
envs of the same policy. No opponent modeling here yet; this is the scaffold
it plugs into.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tom.envs import VizDoomMultiAgentEnv


@dataclass
class PPOConfig:
    total_steps: int = 20_000
    rollout_steps: int = 128
    epochs: int = 3
    minibatch_size: int = 64
    lr: float = 2.5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    log_every_updates: int = 1
    save_path: str | None = None
    save_every_updates: int = 25


class CNNActorCritic(nn.Module):
    def __init__(self, n_actions: int, n_vars: int, hidden: int = 512):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            feat_dim = self.conv(torch.zeros(1, 3, 84, 84)).shape[1]
        self.trunk = nn.Sequential(
            nn.Linear(feat_dim + n_vars, hidden), nn.ReLU(),
        )
        self.pi = nn.Linear(hidden, n_actions)
        self.v = nn.Linear(hidden, 1)

    def forward(self, screen: torch.Tensor, gv: torch.Tensor):
        h = self.conv(screen.float() / 255.0)
        h = torch.cat([h, gv], dim=-1)
        h = self.trunk(h)
        return self.pi(h), self.v(h).squeeze(-1)


def _stack_obs(
    obs: dict[str, dict[str, np.ndarray]],
    agents: list[str],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    screen = np.stack([obs[a]["screen"] for a in agents])
    gv = np.stack([obs[a]["gamevars"] for a in agents])
    return (
        torch.from_numpy(screen).to(device, non_blocking=True),
        torch.from_numpy(gv).to(device, non_blocking=True),
    )


def _compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    next_value: torch.Tensor,
    gamma: float,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    T = rewards.shape[0]
    adv = torch.zeros_like(rewards)
    gae = torch.zeros_like(next_value)
    for t in reversed(range(T)):
        nv = next_value if t == T - 1 else values[t + 1]
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * nv * mask - values[t]
        gae = delta + gamma * lam * mask * gae
        adv[t] = gae
    returns = adv + values
    return adv, returns


def train(cfg: PPOConfig, env_kwargs: dict | None = None, seed: int = 0) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    env_kwargs = env_kwargs or {}
    env = VizDoomMultiAgentEnv(num_players=2, seed=seed, **env_kwargs)

    obs, _ = env.reset()
    agents = list(env.agents)
    n_agents = len(agents)
    n_actions = env.action_space(agents[0]).n
    n_vars = env.n_vars
    H, W = env.frame_shape

    net = CNNActorCritic(n_actions, n_vars).to(device)
    optim = torch.optim.Adam(net.parameters(), lr=cfg.lr, eps=1e-5)

    # rollout buffers shape: (T, n_agents, ...)
    T = cfg.rollout_steps
    buf_screen = torch.zeros((T, n_agents, 3, H, W), dtype=torch.uint8, device=device)
    buf_gv = torch.zeros((T, n_agents, n_vars), dtype=torch.float32, device=device)
    buf_action = torch.zeros((T, n_agents), dtype=torch.long, device=device)
    buf_logp = torch.zeros((T, n_agents), dtype=torch.float32, device=device)
    buf_value = torch.zeros((T, n_agents), dtype=torch.float32, device=device)
    buf_reward = torch.zeros((T, n_agents), dtype=torch.float32, device=device)
    buf_done = torch.zeros((T, n_agents), dtype=torch.float32, device=device)

    ep_returns = {a: 0.0 for a in agents}
    ep_lengths = {a: 0 for a in agents}
    completed_returns: list[float] = []

    global_step = 0
    update_idx = 0
    t0 = time.time()
    print(f"device={device} agents={agents} n_actions={n_actions} n_vars={n_vars}")

    while global_step < cfg.total_steps:
        for t in range(T):
            screen, gv = _stack_obs(obs, agents, device)
            with torch.no_grad():
                logits, values = net(screen, gv)
                dist = torch.distributions.Categorical(logits=logits)
                actions_t = dist.sample()
                logp = dist.log_prob(actions_t)

            buf_screen[t] = screen
            buf_gv[t] = gv
            buf_action[t] = actions_t
            buf_logp[t] = logp
            buf_value[t] = values

            actions_dict = {a: int(actions_t[i].item()) for i, a in enumerate(agents)}
            next_obs, rewards, terms, truncs, infos = env.step(actions_dict)

            rew_vec = torch.tensor(
                [rewards[a] for a in agents], dtype=torch.float32, device=device
            )
            done_vec = torch.tensor(
                [float(terms[a] or truncs[a]) for a in agents],
                dtype=torch.float32,
                device=device,
            )
            buf_reward[t] = rew_vec
            buf_done[t] = done_vec

            for i, a in enumerate(agents):
                ep_returns[a] += rewards[a]
                ep_lengths[a] += 1

            global_step += n_agents

            if all(terms[a] or truncs[a] for a in agents):
                for a in agents:
                    completed_returns.append(ep_returns[a])
                ep_returns = {a: 0.0 for a in agents}
                ep_lengths = {a: 0 for a in agents}
                next_obs, _ = env.reset()
                agents = list(env.agents)

            obs = next_obs

        # bootstrap value
        with torch.no_grad():
            screen, gv = _stack_obs(obs, agents, device)
            _, next_value = net(screen, gv)
        adv, ret = _compute_gae(
            buf_reward, buf_value, buf_done, next_value,
            gamma=cfg.gamma, lam=cfg.gae_lambda,
        )
        adv_flat = adv.reshape(-1)
        ret_flat = ret.reshape(-1)
        adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)

        # flatten (T, n_agents, ...) -> (T*n_agents, ...)
        b_screen = buf_screen.reshape(T * n_agents, 3, H, W)
        b_gv = buf_gv.reshape(T * n_agents, n_vars)
        b_act = buf_action.reshape(-1)
        b_old_logp = buf_logp.reshape(-1)

        N = T * n_agents
        idx = torch.randperm(N, device=device)
        pg_losses: list[float] = []
        v_losses: list[float] = []
        entropies: list[float] = []
        for _ in range(cfg.epochs):
            for start in range(0, N, cfg.minibatch_size):
                mb = idx[start : start + cfg.minibatch_size]
                logits, values = net(b_screen[mb], b_gv[mb])
                dist = torch.distributions.Categorical(logits=logits)
                new_logp = dist.log_prob(b_act[mb])
                ratio = torch.exp(new_logp - b_old_logp[mb])
                mb_adv = adv_flat[mb]
                unclipped = ratio * mb_adv
                clipped = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * mb_adv
                pg = -torch.min(unclipped, clipped).mean()
                v_loss = F.mse_loss(values, ret_flat[mb])
                ent = dist.entropy().mean()
                loss = pg + cfg.vf_coef * v_loss - cfg.ent_coef * ent
                optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
                optim.step()
                pg_losses.append(pg.item())
                v_losses.append(v_loss.item())
                entropies.append(ent.item())

        update_idx += 1
        if cfg.save_path and update_idx % cfg.save_every_updates == 0:
            torch.save(net.state_dict(), cfg.save_path)
        if update_idx % cfg.log_every_updates == 0:
            dt = time.time() - t0
            sps = global_step / max(dt, 1e-6)
            ret_mean = (
                float(np.mean(completed_returns[-20:]))
                if completed_returns
                else float("nan")
            )
            print(
                f"upd={update_idx:4d} step={global_step:7d} "
                f"sps={sps:6.0f} ret20={ret_mean:+.2f} "
                f"pg={np.mean(pg_losses):+.3f} v={np.mean(v_losses):.3f} "
                f"ent={np.mean(entropies):.3f}"
            )

    env.close()
    if cfg.save_path:
        torch.save(net.state_dict(), cfg.save_path)
        print(f"final ckpt saved -> {cfg.save_path}")
    print(f"done — {update_idx} updates, {global_step} env-steps in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--total-steps", type=int, default=20_000)
    ap.add_argument("--rollout-steps", type=int, default=128)
    ap.add_argument("--episode-seconds", type=float, default=720.0)
    ap.add_argument("--ticrate", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    train(
        PPOConfig(total_steps=args.total_steps, rollout_steps=args.rollout_steps),
        env_kwargs=dict(
            episode_timeout_seconds=args.episode_seconds,
            ticrate=args.ticrate,
        ),
        seed=args.seed,
    )
