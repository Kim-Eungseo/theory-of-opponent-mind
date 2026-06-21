"""Opponent pool: a collection of partner factories with sampling.

A factory is a *callable that returns a fresh ``Partner`` instance* —
this matters because (a) we want each parallel env to have an
independent partner state, and (b) some partners (RandomPartner,
WanderingPartner) carry internal RNG state that must not be shared
across envs / episodes.

We expose three pools by default:
    * train_pool      — used during Phase-2 online training (sampled per episode)
    * held_out_pool   — never seen during training; used for evaluation
    * baseline_pool   — fixed (NOOP / Random / Direction) reference partners

Each partner carries a string ``name`` used as its opponent-id for the
contrastive loss and the continual evaluation protocol.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Sequence

from tom.opponent_pool.partners import (
    DirectionPartner,
    MixedPartner,
    NoopPartner,
    Partner,
    RandomPartner,
    WanderingPartner,
)

PartnerFactory = Callable[[], Partner]


@dataclass
class OpponentPool:
    factories: list[PartnerFactory] = field(default_factory=list)
    names: list[str] = field(default_factory=list)
    _rng: random.Random = field(default_factory=lambda: random.Random())

    def __post_init__(self):
        if len(self.factories) != len(self.names):
            raise ValueError("factories and names must have same length")

    def __len__(self) -> int:
        return len(self.factories)

    def add(self, factory: PartnerFactory, name: str) -> None:
        self.factories.append(factory)
        self.names.append(name)

    def sample_idx(self) -> int:
        return self._rng.randrange(len(self.factories))

    def sample(self) -> tuple[Partner, str, int]:
        i = self.sample_idx()
        return self.factories[i](), self.names[i], i

    def get(self, idx: int) -> tuple[Partner, str]:
        return self.factories[idx](), self.names[idx]

    def seed(self, s: int) -> None:
        self._rng = random.Random(s)


def make_default_train_pool(seed_base: int = 0) -> OpponentPool:
    """A small scripted-partner pool good enough for Phase-2 smoke training."""
    pool = OpponentPool()
    pool.add(lambda: NoopPartner(), "noop")
    for d, dname in enumerate(["N", "S", "E", "W"]):
        pool.add(lambda d=d, dn=dname: DirectionPartner(d, name=f"dir_{dn}"), f"dir_{dname}")
    for k in range(3):
        pool.add(lambda k=k: WanderingPartner(seed=seed_base + 1000 * k, name=f"wander_{k}"),
                 f"wander_{k}")
    for k in range(2):
        pool.add(lambda k=k: RandomPartner(seed=seed_base + 2000 * k, name=f"random_{k}"),
                 f"random_{k}")
    pool.seed(seed_base)
    return pool


def make_default_heldout_pool(seed_base: int = 9999) -> OpponentPool:
    """Held-out partners — disjoint from train pool."""
    pool = OpponentPool()
    # different RNG seeds + slightly different parameters than train pool
    pool.add(lambda: WanderingPartner(momentum=0.5, seed=seed_base, name="wander_held_a"),
             "wander_held_a")
    pool.add(lambda: WanderingPartner(momentum=0.9, seed=seed_base + 1, name="wander_held_b"),
             "wander_held_b")
    pool.add(lambda: RandomPartner(seed=seed_base + 2, name="random_held"),
             "random_held")
    pool.add(lambda: DirectionPartner(0, stay_prob=0.4, name="dir_N_held"),
             "dir_N_held")
    pool.seed(seed_base)
    return pool


def add_checkpoint_partners(
    pool: OpponentPool,
    ckpts: Sequence[str],
    partner_slot: str = "agent_1",
    device: str = "cpu",
) -> OpponentPool:
    """Register trained-IPPO checkpoint partners into a pool."""
    from tom.opponent_pool.wrapped_agent import CheckpointPartner
    for p in ckpts:
        # snapshot args to avoid late-binding loop bug
        pool.add(
            lambda p=p, ps=partner_slot, dv=device: CheckpointPartner(
                p, partner_slot=ps, device=dv, name=f"ckpt:{p}",
            ),
            f"ckpt:{p}",
        )
    return pool
