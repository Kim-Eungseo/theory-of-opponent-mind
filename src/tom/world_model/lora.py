"""LoRA adapter for Phase-2 fine-tuning of Phase-1 encoder weights.

Wraps every ``nn.Linear`` in the given module with a low-rank update
``y = W·x + (α/r) · B·A·x``. The base weight is frozen; only A (r × in)
and B (out × r) train. Typical choice: r=8, α=16 → 16 + (r·in + out·r) ≈
2-3% of original Linear parameters.
"""
from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.r = r
        self.scale = alpha / r
        self.lora_A = nn.Linear(base.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, base.out_features, bias=False)
        nn.init.normal_(self.lora_A.weight, std=0.02)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scale * self.lora_B(self.lora_A(x))


def apply_lora(module: nn.Module, r: int = 8, alpha: float = 16.0,
               include_names: Iterable[str] | None = None) -> nn.Module:
    """In-place replace every nn.Linear (optionally only those in
    ``include_names``) with a LoRALinear. Returns the modified module.
    """
    for child_name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            if include_names is None or child_name in include_names:
                setattr(module, child_name, LoRALinear(child, r=r, alpha=alpha))
        else:
            apply_lora(child, r=r, alpha=alpha, include_names=include_names)
    return module


def lora_parameters(module: nn.Module) -> list[nn.Parameter]:
    """Collect only the LoRA-trainable parameters in a module tree."""
    out: list[nn.Parameter] = []
    for sub in module.modules():
        if isinstance(sub, LoRALinear):
            out += list(sub.lora_A.parameters())
            out += list(sub.lora_B.parameters())
    return out


def freeze_non_lora(module: nn.Module) -> None:
    """Freeze every parameter that isn't part of a LoRALinear adapter."""
    for sub in module.modules():
        if isinstance(sub, LoRALinear):
            for p in sub.lora_A.parameters():
                p.requires_grad = True
            for p in sub.lora_B.parameters():
                p.requires_grad = True
            for p in sub.base.parameters():
                p.requires_grad = False
        else:
            for name, p in sub.named_parameters(recurse=False):
                # leave LoRA params (handled above) alone, freeze the rest
                p.requires_grad = False
