"""Opponent / partner agents for the two-phase pipeline.

Partners are *scripted or learned* policies that the learner interacts with.
For Phase 1 (world model training) we use simple partners (NOOP, Random,
Direction); for Phase 2 we sample from a curated opponent pool of both
scripted and trained agents.
"""
from tom.opponent_pool.partners import (
    Partner,
    NoopPartner,
    RandomPartner,
    DirectionPartner,
    WanderingPartner,
    MixedPartner,
)
from tom.opponent_pool.wrapped_agent import CheckpointPartner
from tom.opponent_pool.pool import OpponentPool

__all__ = [
    "Partner",
    "NoopPartner",
    "RandomPartner",
    "DirectionPartner",
    "WanderingPartner",
    "MixedPartner",
    "CheckpointPartner",
    "OpponentPool",
]
