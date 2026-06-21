"""Contrastive learning of opponent latents (MoCo-style InfoNCE).

Two trajectory chunks from the *same* opponent should map to similar
latents; chunks from *different* opponents should map far apart. We use
a momentum target encoder for stable keys.

Typical use::

    enc = TrajectoryEncoder(...)
    mom = MomentumEncoder(enc, momentum=0.99)
    # forward
    q  = mom.encode_query(obs_q,  act_q)   # gradient-tracked
    k  = mom.encode_key  (obs_k,  act_k)   # no_grad, EMA encoder
    loss = info_nce(q, k_pos, k_negs, temperature=0.1)
    mom.update_key()

Negative banks: either in-batch negatives or a momentum queue.
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class MomentumEncoder(nn.Module):
    """Wraps a base encoder, maintains an EMA target encoder.

    The base (``self.q``) is the trainable encoder; ``self.k`` is the
    EMA target used to produce keys without gradient.
    """

    def __init__(self, encoder: nn.Module, momentum: float = 0.99):
        super().__init__()
        self.q = encoder
        self.k = copy.deepcopy(encoder)
        for p in self.k.parameters():
            p.requires_grad = False
        self.m = float(momentum)

    @torch.no_grad()
    def update_key(self) -> None:
        for q_p, k_p in zip(self.q.parameters(), self.k.parameters()):
            k_p.data.mul_(self.m).add_(q_p.data, alpha=1.0 - self.m)

    def encode_query(self, *args, **kwargs) -> torch.Tensor:
        return self.q(*args, **kwargs)

    @torch.no_grad()
    def encode_key(self, *args, **kwargs) -> torch.Tensor:
        return self.k(*args, **kwargs)


def info_nce_loss(
    q: torch.Tensor,
    k_pos: torch.Tensor,
    k_negs: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """
    q:      (B, D) query embeddings
    k_pos:  (B, D) positive keys (e.g. another chunk of the same opponent)
    k_negs: (B, K, D) negative keys
    """
    q = F.normalize(q, dim=-1)
    k_pos = F.normalize(k_pos, dim=-1)
    k_negs = F.normalize(k_negs, dim=-1)
    pos = (q * k_pos).sum(-1, keepdim=True) / temperature       # (B, 1)
    neg = torch.einsum("bd,bkd->bk", q, k_negs) / temperature   # (B, K)
    logits = torch.cat([pos, neg], dim=1)                       # (B, 1+K)
    labels = torch.zeros(q.size(0), dtype=torch.long, device=q.device)
    return F.cross_entropy(logits, labels)


def in_batch_contrastive(
    q: torch.Tensor,           # (B, D)
    k_pos: torch.Tensor,       # (B, D) positive paired with each q
    opponent_ids: torch.Tensor,  # (B,) — same id = positive, different = negative
    temperature: float = 0.1,
) -> torch.Tensor:
    """Symmetric InfoNCE using same-batch keys, masking out same-opponent
    rows from the negative pool. ``opponent_ids[i] == opponent_ids[j]``
    means row j is a positive for row i.
    """
    B = q.size(0)
    q = F.normalize(q, dim=-1)
    k = F.normalize(k_pos, dim=-1)
    sim = q @ k.t() / temperature  # (B, B)
    # build a label mask: each row i has 1 in position i (its paired positive)
    labels = torch.arange(B, device=q.device)
    # mask out other same-opponent rows from negatives so they don't compete
    same_opp = opponent_ids[:, None] == opponent_ids[None, :]
    same_opp.fill_diagonal_(False)
    sim = sim.masked_fill(same_opp, float("-inf"))
    return F.cross_entropy(sim, labels)
