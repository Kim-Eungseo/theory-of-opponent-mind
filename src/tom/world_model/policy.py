"""Conditional actor-critic: π(a | h, z_opp), V(h, z_opp).

The encoder ``h = φ(o)`` comes from the (LoRA-fine-tuned) Phase-1 world
model encoder. ``z_opp`` is the latent emitted by the trajectory encoder
η for the current opponent. An optional auxiliary head ``π̂_opp``
predicts the partner's *next* action — same routing pattern as our
earlier TOM+BAD trainer, but on top of the learned opponent latent.
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


class ConditionalActorCritic(nn.Module):
    """Heads on top of (h, z_opp). The encoder φ is owned externally."""

    def __init__(
        self,
        latent_dim: int,
        opp_dim: int,
        n_actions: int,
        hidden: int = 256,
        with_om_head: bool = True,
    ):
        super().__init__()
        in_dim = latent_dim + opp_dim
        self.with_om_head = with_om_head

        def mlp(out_dim: int, head_gain: float = np.sqrt(2)) -> nn.Sequential:
            l1 = nn.Linear(in_dim, hidden); _orth(l1)
            l2 = nn.Linear(hidden, hidden); _orth(l2)
            l3 = nn.Linear(hidden, out_dim); _orth(l3, gain=head_gain)
            return nn.Sequential(l1, nn.Tanh(), l2, nn.Tanh(), l3)

        self.policy = mlp(n_actions, head_gain=0.01)
        self.value = mlp(1, head_gain=1.0)
        self.om_head = mlp(n_actions, head_gain=0.1) if with_om_head else None

    def _features(self, h: torch.Tensor, z_opp: torch.Tensor) -> torch.Tensor:
        return torch.cat([h, z_opp], dim=-1)

    def act(self, h: torch.Tensor, z_opp: torch.Tensor, deterministic: bool = False):
        x = self._features(h, z_opp)
        logits = self.policy(x)
        v = self.value(x).squeeze(-1)
        dist = torch.distributions.Categorical(logits=logits)
        a = logits.argmax(-1) if deterministic else dist.sample()
        return a, dist.log_prob(a), v

    def evaluate(self, h: torch.Tensor, z_opp: torch.Tensor, action: torch.Tensor):
        x = self._features(h, z_opp)
        logits = self.policy(x)
        v = self.value(x).squeeze(-1)
        dist = torch.distributions.Categorical(logits=logits)
        logp = dist.log_prob(action)
        ent = dist.entropy()
        om_logits = self.om_head(x) if self.om_head is not None else None
        return logp, ent, v, om_logits

    def value_only(self, h: torch.Tensor, z_opp: torch.Tensor) -> torch.Tensor:
        x = self._features(h, z_opp)
        return self.value(x).squeeze(-1)
