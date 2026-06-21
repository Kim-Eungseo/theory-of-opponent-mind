"""Two-phase OnCOM (Online Continual Opponent Modeling) pipeline.

Phase 1 — train a world model in solo Overcooked (scripted/random partner):
    encoder φ(o) → h
    dynamics ψ(h, a_own, a_opp) → ĥ'
    reward   r̂(h, a_own, a_opp)

Phase 2 — online continual opponent modeling with a learned opponent pool:
    LoRA-fine-tuned encoder φ̃
    trajectory encoder η(τ_opp) → z_opp  (contrastive)
    conditional policy π(a | h, z_opp)   (PPO)
    aux OM head π̂_opp(a^opp | h, z_opp)  (cross-entropy)
"""
from tom.world_model.model import (
    GridEncoder,
    DynamicsHead,
    RewardHead,
    WorldModel,
)
from tom.world_model.lora import LoRALinear, apply_lora
from tom.world_model.trajectory_encoder import TrajectoryEncoder
from tom.world_model.contrastive import MomentumEncoder, info_nce_loss
from tom.world_model.policy import ConditionalActorCritic

__all__ = [
    "GridEncoder", "DynamicsHead", "RewardHead", "WorldModel",
    "LoRALinear", "apply_lora",
    "TrajectoryEncoder",
    "MomentumEncoder", "info_nce_loss",
    "ConditionalActorCritic",
]
