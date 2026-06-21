"""Load a saved IPPO partner checkpoint and serve it as a fixed partner.

The checkpoint must have been produced by ``ippo_overcooked.py`` (which
stores both agents' state dicts together with the cfg dict). Obs-dim
*must* match what the learner sees in the solo env — typically the
partner ckpt should be trained at the same ``view_radius`` as Phase-2.
"""
from __future__ import annotations

import numpy as np
import torch


class CheckpointPartner:
    """Trained partner from an ``ippo_overcooked`` checkpoint.

    Args:
        ckpt_path: path to ``ckpt_*.pt``
        partner_slot: which slot to load — typically ``"agent_1"``
        device: cpu/cuda
        deterministic: if True, argmax action; else sample.
    """

    def __init__(
        self,
        ckpt_path: str,
        partner_slot: str = "agent_1",
        device: str | torch.device = "cpu",
        deterministic: bool = False,
        name: str | None = None,
    ):
        from tom.training.ippo_overcooked import ActorCriticOM

        self.ckpt_path = str(ckpt_path)
        self.device = torch.device(device)
        self.deterministic = bool(deterministic)
        ck = torch.load(self.ckpt_path, map_location=self.device)
        cfg = ck["cfg"]

        # Infer obs_dim from the saved encoder's first Linear layer
        state_dict = ck["nets"][partner_slot]
        first_w_key = next(k for k in state_dict if k.startswith("encoder.0.weight"))
        obs_dim = int(state_dict[first_w_key].shape[1])
        n_actions = 6  # Overcooked default
        self.obs_dim = obs_dim
        self.n_actions = n_actions

        # rebuild network with the cfg-recorded knobs
        use_tom = (cfg.get("tom_coef", 0.0) > 0) or cfg.get("tom_in_policy", False)
        net = ActorCriticOM(
            obs_dim=obs_dim,
            n_actions=n_actions,
            hidden=cfg.get("hidden", 256),
            om_in_policy=cfg.get("om_in_policy", False),
            use_tom=use_tom,
            tom_hidden=cfg.get("tom_hidden", 128),
            tom_in_policy=cfg.get("tom_in_policy", False),
        ).to(self.device)
        # tolerate missing/extra keys (e.g., tom_head absent when partner had no TOM)
        net.load_state_dict(state_dict, strict=False)
        net.eval()
        for p in net.parameters():
            p.requires_grad = False
        self.net = net
        self._uses_tom_in_policy = cfg.get("tom_in_policy", False)
        # partner_hist required only if tom_in_policy=True; we maintain a tiny ring buffer
        self._K = int(cfg.get("tom_history_len", 8))
        self._hist: np.ndarray | None = None
        self.name = name or f"ckpt:{self.ckpt_path}"

    def reset(self) -> None:
        if self._uses_tom_in_policy:
            self._hist = np.zeros((self._K, self.obs_dim), dtype=np.float32)
        else:
            self._hist = None

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> int:
        if obs.shape[-1] != self.obs_dim:
            raise ValueError(
                f"CheckpointPartner obs_dim mismatch: ckpt expects {self.obs_dim}, got {obs.shape[-1]}"
            )
        o = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        partner_hist = None
        if self._uses_tom_in_policy:
            # The partner here is acting BLINDLY — it doesn't get its own partner's
            # trajectory in a solo-env wrapper. Pass zero history.
            self._hist = np.roll(self._hist, -1, axis=0)
            self._hist[-1] = obs
            partner_hist = torch.as_tensor(self._hist, dtype=torch.float32,
                                           device=self.device).unsqueeze(0)
        a, _, _ = self.net.act(o, partner_hist=partner_hist)
        return int(a.item())
