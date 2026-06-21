"""World model: encoder + (joint-action conditioned) dynamics + reward.

The dynamics is *factorized over both agents' actions* — the world is
opponent-invariant in the sense that ``P(s'|s, a_own, a_opp)`` is a fixed
function of the joint action. The opponent's *policy* changes between
encounters; the world dynamics function does not. Phase 1 trains this
fixed dynamics from solo/random-partner trajectories.

Latent-target dynamics loss (no pixel decoder by default):
    L_dyn = ||ψ(φ(s_t), a_own, a_opp) - sg(φ(s_{t+1}))||²

A small reward head shares the latent. Optional decoder reconstruction is
available as an ablation knob (``decode=True``).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _orth(m: nn.Linear, gain: float = np.sqrt(2)):
    nn.init.orthogonal_(m.weight, gain=gain)
    if m.bias is not None:
        nn.init.zeros_(m.bias)


class GridEncoder(nn.Module):
    """Flattened-grid → latent. Reused as φ in both phases."""

    def __init__(self, obs_dim: int, hidden: int = 256, latent: int = 128, layers: int = 2):
        super().__init__()
        mods: list[nn.Module] = []
        prev = obs_dim
        for _ in range(layers):
            lin = nn.Linear(prev, hidden)
            _orth(lin)
            mods += [lin, nn.ReLU()]
            prev = hidden
        head = nn.Linear(hidden, latent)
        _orth(head, gain=1.0)
        mods.append(head)
        self.net = nn.Sequential(*mods)
        self.out_dim = latent

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


def _action_pair_features(a_own: torch.Tensor, a_opp: torch.Tensor, n_actions: int) -> torch.Tensor:
    own = F.one_hot(a_own.long(), n_actions).float()
    opp = F.one_hot(a_opp.long(), n_actions).float()
    return torch.cat([own, opp], dim=-1)


class DynamicsHead(nn.Module):
    """ψ: (h, a_own, a_opp) → ĥ'."""

    def __init__(self, latent: int = 128, n_actions: int = 6, hidden: int = 256, layers: int = 2):
        super().__init__()
        self.n_actions = n_actions
        in_dim = latent + 2 * n_actions
        mods: list[nn.Module] = []
        prev = in_dim
        for _ in range(layers):
            lin = nn.Linear(prev, hidden)
            _orth(lin)
            mods += [lin, nn.ReLU()]
            prev = hidden
        out = nn.Linear(hidden, latent)
        _orth(out, gain=1.0)
        mods.append(out)
        self.net = nn.Sequential(*mods)

    def forward(self, h: torch.Tensor, a_own: torch.Tensor, a_opp: torch.Tensor) -> torch.Tensor:
        x = torch.cat([h, _action_pair_features(a_own, a_opp, self.n_actions)], dim=-1)
        return self.net(x)


class RewardHead(nn.Module):
    """r̂: (h, a_own, a_opp) → ŕ."""

    def __init__(self, latent: int = 128, n_actions: int = 6, hidden: int = 128):
        super().__init__()
        self.n_actions = n_actions
        in_dim = latent + 2 * n_actions
        lin1 = nn.Linear(in_dim, hidden); _orth(lin1)
        lin2 = nn.Linear(hidden, 1); _orth(lin2, gain=1.0)
        self.net = nn.Sequential(lin1, nn.ReLU(), lin2)

    def forward(self, h: torch.Tensor, a_own: torch.Tensor, a_opp: torch.Tensor) -> torch.Tensor:
        x = torch.cat([h, _action_pair_features(a_own, a_opp, self.n_actions)], dim=-1)
        return self.net(x).squeeze(-1)


class Decoder(nn.Module):
    """g: h → ô  (optional reconstruction head)."""

    def __init__(self, obs_dim: int, latent: int = 128, hidden: int = 256, layers: int = 2):
        super().__init__()
        mods: list[nn.Module] = []
        prev = latent
        for _ in range(layers):
            lin = nn.Linear(prev, hidden); _orth(lin)
            mods += [lin, nn.ReLU()]
            prev = hidden
        out = nn.Linear(hidden, obs_dim); _orth(out, gain=1.0)
        mods.append(out)
        self.net = nn.Sequential(*mods)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


class WorldModel(nn.Module):
    """Full world model: encoder + dynamics + reward (+ optional decoder)."""

    def __init__(
        self,
        obs_dim: int,
        n_actions: int = 6,
        latent: int = 128,
        hidden: int = 256,
        decode: bool = False,
        decoder_weight: float = 0.1,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.latent_dim = latent
        self.encoder = GridEncoder(obs_dim, hidden=hidden, latent=latent)
        self.dynamics = DynamicsHead(latent, n_actions, hidden=hidden)
        self.reward = RewardHead(latent, n_actions, hidden=hidden)
        self.decoder: Decoder | None = Decoder(obs_dim, latent, hidden=hidden) if decode else None
        self.decoder_weight = float(decoder_weight)

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs)

    def forward(self, obs, a_own, a_opp):
        h = self.encode(obs)
        h_next_pred = self.dynamics(h, a_own, a_opp)
        r_pred = self.reward(h, a_own, a_opp)
        recon = self.decoder(h) if self.decoder is not None else None
        return h, h_next_pred, r_pred, recon

    def loss(self, obs, a_own, a_opp, next_obs, reward) -> tuple[torch.Tensor, dict]:
        h, h_next_pred, r_pred, recon = self(obs, a_own, a_opp)
        with torch.no_grad():
            h_next_target = self.encode(next_obs)
        l_dyn = F.mse_loss(h_next_pred, h_next_target)
        l_rew = F.mse_loss(r_pred, reward)
        total = l_dyn + l_rew
        logs: dict[str, float] = {"l_dyn": l_dyn.item(), "l_rew": l_rew.item()}
        if self.decoder is not None and recon is not None:
            l_rec = F.mse_loss(recon, obs)
            total = total + self.decoder_weight * l_rec
            logs["l_rec"] = l_rec.item()
        return total, logs
