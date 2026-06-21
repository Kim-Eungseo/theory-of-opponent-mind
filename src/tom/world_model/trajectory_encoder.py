"""η: opponent trajectory → opponent latent z_opp.

We embed the partner's recent (observed-from-our-view) (obs, action) pairs
through a GRU and emit a fixed-size latent. The latent is trained
contrastively (InfoNCE) so trajectories from the same opponent map near
each other; on top of that the policy and the OM aux head consume it as
an extra input.

Input shapes:
    obs_seq:    (B, T, obs_dim)
    action_seq: (B, T)   (partner's action ids)

Output: z_opp ∈ ℝ^{latent_dim}.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TrajectoryEncoder(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        obs_embed_dim: int = 64,
        act_embed_dim: int = 16,
        hidden: int = 128,
        latent_dim: int = 32,
        gru_layers: int = 1,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.latent_dim = latent_dim

        self.obs_embed = nn.Linear(obs_dim, obs_embed_dim)
        self.action_embed = nn.Embedding(n_actions, act_embed_dim)
        self.gru = nn.GRU(
            input_size=obs_embed_dim + act_embed_dim,
            hidden_size=hidden,
            num_layers=gru_layers,
            batch_first=True,
        )
        self.head = nn.Linear(hidden, latent_dim)

        nn.init.orthogonal_(self.obs_embed.weight)
        nn.init.zeros_(self.obs_embed.bias)
        nn.init.orthogonal_(self.head.weight, gain=1.0)
        nn.init.zeros_(self.head.bias)

    def forward(self, obs_seq: torch.Tensor, action_seq: torch.Tensor) -> torch.Tensor:
        """(B, T, obs_dim), (B, T) → (B, latent_dim)."""
        o = self.obs_embed(obs_seq)
        a = self.action_embed(action_seq.long())
        x = torch.cat([o, a], dim=-1)
        out, _ = self.gru(x)
        last = out[:, -1, :]
        return self.head(last)

    def empty_latent(self, batch: int, device) -> torch.Tensor:
        """Zero latent — used at episode start before any partner step is observed."""
        return torch.zeros(batch, self.latent_dim, device=device)
