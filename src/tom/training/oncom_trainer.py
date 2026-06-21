"""Phase-2: Online Continual Opponent Modeling (OnCOM).

A single learner agent interacts with partners sampled from an opponent
pool. Per rollout:

* Maintain a per-env ring of the partner's last K (obs, action) pairs.
* Trajectory encoder η emits ``z_opp`` from that ring.
* Conditional policy π(a | φ(o), z_opp) picks the learner's action.
* Aux OM head π̂_opp(a^opp | φ(o), z_opp) is supervised by the partner's
  observed action.
* Trajectory encoder is trained contrastively (InfoNCE) so chunks from
  the same opponent map closer than chunks from different opponents.

Encoder transfer modes:
    * ``frozen``   — Phase-1 weights, no updates.
    * ``lora``     — Phase-1 weights frozen, low-rank deltas trained.
    * ``full``     — Phase-1 weights as init, fully trainable.
"""
from __future__ import annotations

import copy
import os
import time
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from tom.envs.overcooked_solo import VecOvercookedSoloEnv
from tom.opponent_pool.pool import OpponentPool
from tom.world_model.contrastive import (
    MomentumEncoder,
    in_batch_contrastive,
    info_nce_loss,
)
from tom.world_model.lora import apply_lora, freeze_non_lora, lora_parameters
from tom.world_model.model import GridEncoder, WorldModel
from tom.world_model.policy import ConditionalActorCritic
from tom.world_model.trajectory_encoder import TrajectoryEncoder


# ---- GAE ----------------------------------------------------------------

def compute_gae(rewards, values, dones, last_values, gamma=0.99, lam=0.98):
    T, N = rewards.shape
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros(N, device=rewards.device)
    for t in reversed(range(T)):
        next_v = last_values if t == T - 1 else values[t + 1]
        nonterm = 1.0 - dones[t].float()
        delta = rewards[t] + gamma * next_v * nonterm - values[t]
        gae = delta + gamma * lam * nonterm * gae
        advantages[t] = gae
    returns = advantages + values
    return advantages, returns


# ---- Config -------------------------------------------------------------

@dataclass
class OnComConfig:
    layout: str = "asymmetric_advantages"
    horizon: int = 400
    view_radius: int = 2
    num_envs: int = 16
    total_steps: int = 5_000_000
    rollout_steps: int = 400

    # encoder transfer
    wm_ckpt: str | None = None             # Phase-1 world model checkpoint
    encoder_mode: str = "lora"             # frozen | lora | full
    lora_r: int = 8
    lora_alpha: float = 16.0
    latent_dim: int = 128                  # must match Phase-1 latent
    hidden: int = 256

    # trajectory encoder
    traj_K: int = 16
    z_dim: int = 32
    traj_hidden: int = 128
    traj_obs_embed: int = 64
    traj_act_embed: int = 16

    # PPO
    lr_policy: float = 3e-4
    lr_encoder: float = 1e-4
    lr_traj: float = 3e-4
    learning_epochs: int = 4
    mini_batches: int = 4
    clip_eps: float = 0.05
    ent_coef: float = 0.1
    vf_coef: float = 0.5
    gamma: float = 0.99
    gae_lambda: float = 0.98
    max_grad_norm: float = 0.5

    # aux losses
    om_coef: float = 0.5                   # CE(π̂_opp(h, z_opp), a^opp)
    contrastive_coef: float = 0.5
    contrastive_temperature: float = 0.1
    momentum: float = 0.99
    contrastive_chunk: int = 16            # length of trajectory chunk for InfoNCE

    # reward shaping
    shaped_reward_coef_start: float = 1.0
    shaped_reward_coef_end: float = 0.0
    shaped_reward_anneal_frac: float = 0.5

    log_dir: str = "runs_oncom/oncom"
    log_interval: int = 1
    ckpt_interval_steps: int = 500_000
    seed: int = 0
    device: str = "auto"


# ---- Trainer ------------------------------------------------------------

def _t(x, device, dtype=torch.float32):
    return torch.as_tensor(x, dtype=dtype, device=device)


def _build_encoder(cfg: OnComConfig, obs_dim: int, device) -> tuple[nn.Module, list[nn.Parameter]]:
    """Construct φ encoder and return its trainable parameter list."""
    if cfg.wm_ckpt is not None:
        ck = torch.load(cfg.wm_ckpt, map_location=device)
        wm_obs_dim = int(ck["obs_dim"])
        if wm_obs_dim != obs_dim:
            raise ValueError(
                f"World-model obs_dim={wm_obs_dim} does not match env obs_dim={obs_dim}"
            )
        wm = WorldModel(
            obs_dim=obs_dim,
            n_actions=int(ck["n_actions"]),
            latent=cfg.latent_dim,
            hidden=cfg.hidden,
        )
        wm.load_state_dict(ck["model"], strict=False)
        encoder = wm.encoder
    else:
        encoder = GridEncoder(obs_dim, hidden=cfg.hidden, latent=cfg.latent_dim)
    encoder = encoder.to(device)

    mode = cfg.encoder_mode
    if mode == "frozen":
        for p in encoder.parameters():
            p.requires_grad = False
        trainables: list[nn.Parameter] = []
    elif mode == "lora":
        apply_lora(encoder, r=cfg.lora_r, alpha=cfg.lora_alpha)
        encoder = encoder.to(device)
        freeze_non_lora(encoder)
        trainables = lora_parameters(encoder)
    elif mode == "full":
        for p in encoder.parameters():
            p.requires_grad = True
        trainables = list(encoder.parameters())
    else:
        raise ValueError(f"unknown encoder_mode {mode}")
    return encoder, trainables


class OnComLearner(nn.Module):
    """Bundle encoder + trajectory encoder + conditional policy."""

    def __init__(self, encoder: nn.Module, cfg: OnComConfig, obs_dim: int, n_actions: int):
        super().__init__()
        self.encoder = encoder
        self.traj = TrajectoryEncoder(
            obs_dim=obs_dim,
            n_actions=n_actions,
            obs_embed_dim=cfg.traj_obs_embed,
            act_embed_dim=cfg.traj_act_embed,
            hidden=cfg.traj_hidden,
            latent_dim=cfg.z_dim,
        )
        self.policy = ConditionalActorCritic(
            latent_dim=cfg.latent_dim,
            opp_dim=cfg.z_dim,
            n_actions=n_actions,
            hidden=cfg.hidden,
        )

    @torch.no_grad()
    def act(self, obs, hist_obs, hist_act, deterministic=False):
        h = self.encoder(obs)
        z = self.traj(hist_obs, hist_act)
        a, logp, v = self.policy.act(h, z, deterministic=deterministic)
        return a, logp, v, z

    def evaluate(self, obs, hist_obs, hist_act, action):
        h = self.encoder(obs)
        z = self.traj(hist_obs, hist_act)
        logp, ent, v, om_logits = self.policy.evaluate(h, z, action)
        return logp, ent, v, om_logits, z


# ---- Main loop ----------------------------------------------------------

def train_oncom(
    cfg: OnComConfig,
    train_pool: OpponentPool,
    eval_pool: OpponentPool | None = None,
) -> str:
    if cfg.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.device)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    train_pool.seed(cfg.seed)

    # ---- env ----
    def _partner_factory(env_idx: int):
        # initial partner — replaced each episode in the rollout loop
        partner, _, _ = train_pool.sample()
        return partner

    env = VecOvercookedSoloEnv(
        num_envs=cfg.num_envs,
        partner_factory=_partner_factory,
        layout=cfg.layout,
        horizon=cfg.horizon,
        shaped_reward_coef=cfg.shaped_reward_coef_start,
        view_radius=cfg.view_radius,
        learner_idx=0,
        seed=cfg.seed,
    )
    obs_dim = env.obs_dim
    n_actions = env.n_actions

    # ---- encoder + learner ----
    encoder, enc_train_params = _build_encoder(cfg, obs_dim, device)
    learner = OnComLearner(encoder, cfg, obs_dim, n_actions).to(device)

    # InfoNCE: momentum target for the trajectory encoder
    traj_mom = MomentumEncoder(learner.traj, momentum=cfg.momentum).to(device)

    # parameter groups
    enc_params = enc_train_params
    traj_params = list(learner.traj.parameters())
    policy_params = list(learner.policy.parameters())
    optim_groups = []
    if enc_params:
        optim_groups.append({"params": enc_params, "lr": cfg.lr_encoder})
    optim_groups.append({"params": traj_params, "lr": cfg.lr_traj})
    optim_groups.append({"params": policy_params, "lr": cfg.lr_policy})
    opt = torch.optim.Adam(optim_groups)

    os.makedirs(cfg.log_dir, exist_ok=True)
    writer = SummaryWriter(cfg.log_dir)
    print(
        f"[oncom] obs={obs_dim}  n_act={n_actions}  enc_mode={cfg.encoder_mode}  "
        f"wm_ckpt={'yes' if cfg.wm_ckpt else 'no'}  pool_size={len(train_pool)}"
    )

    # per-env state
    N = cfg.num_envs
    K = cfg.traj_K
    # partner history rings (zero-padded at episode start)
    hist_obs = np.zeros((N, K, obs_dim), dtype=np.float32)
    hist_act = np.zeros((N, K), dtype=np.int64)
    cur_opp_idx = np.zeros(N, dtype=np.int64)
    cur_opp_name = ["" for _ in range(N)]

    def _assign_new_partners():
        for i in range(N):
            p, name, idx = train_pool.sample()
            env.set_partner_for(i, p)
            cur_opp_idx[i] = idx
            cur_opp_name[i] = name

    _assign_new_partners()
    obs, _ = env.reset(seed=cfg.seed)
    hist_obs[:] = 0.0
    hist_act[:] = 0

    # per-env episode return accumulators
    ep_return_running = np.zeros(N, dtype=np.float32)
    ep_len_running = np.zeros(N, dtype=np.int32)

    rolling_sparse: list[float] = []
    rolling_shaped: list[float] = []
    rolling_len: list[int] = []
    global_step = 0
    iterations = max(1, cfg.total_steps // (cfg.rollout_steps * N) + 1)
    t0 = time.time()

    for it in range(iterations):
        # ---- shaping anneal ----
        frac = max(0.0, min(1.0, global_step / max(1, int(cfg.total_steps * cfg.shaped_reward_anneal_frac))))
        shaped_coef = cfg.shaped_reward_coef_start + frac * (cfg.shaped_reward_coef_end - cfg.shaped_reward_coef_start)
        for e in env.envs:
            e.base.shaped_reward_coef = shaped_coef

        # ---- rollout ----
        T = cfg.rollout_steps
        buf_obs = torch.zeros(T, N, obs_dim, device=device)
        buf_act = torch.zeros(T, N, dtype=torch.long, device=device)
        buf_logp = torch.zeros(T, N, device=device)
        buf_val = torch.zeros(T, N, device=device)
        buf_rew = torch.zeros(T, N, device=device)
        buf_done = torch.zeros(T, N, device=device)
        buf_hist_obs = torch.zeros(T, N, K, obs_dim, device=device)
        buf_hist_act = torch.zeros(T, N, K, dtype=torch.long, device=device)
        buf_p_act = torch.zeros(T, N, dtype=torch.long, device=device)
        buf_opp_idx = torch.zeros(T, N, dtype=torch.long, device=device)

        for t in range(T):
            obs_t = _t(obs, device)
            hist_obs_t = _t(hist_obs, device)
            hist_act_t = _t(hist_act, device, dtype=torch.long)

            with torch.no_grad():
                a, logp, v, _ = learner.act(obs_t, hist_obs_t, hist_act_t)
            a_np = a.cpu().numpy().astype(np.int64)
            buf_obs[t] = obs_t
            buf_act[t] = a
            buf_logp[t] = logp
            buf_val[t] = v
            buf_hist_obs[t] = hist_obs_t
            buf_hist_act[t] = hist_act_t
            buf_opp_idx[t] = torch.as_tensor(cur_opp_idx, device=device)

            new_obs, rew, term, trunc, info_list = env.step(a_np)
            global_step += N
            buf_rew[t] = _t(rew, device)
            done_mask = np.array([term[i] or trunc[i] for i in range(N)], dtype=np.float32)
            buf_done[t] = _t(done_mask, device)

            # partner data — update history rings
            partner_obs_np = np.stack([info_list[i]["partner_obs"] for i in range(N)])
            partner_act_np = np.array([info_list[i]["partner_action"] for i in range(N)], dtype=np.int64)
            buf_p_act[t] = torch.as_tensor(partner_act_np, device=device)

            hist_obs = np.roll(hist_obs, -1, axis=1)
            hist_obs[:, -1, :] = partner_obs_np
            hist_act = np.roll(hist_act, -1, axis=1)
            hist_act[:, -1] = partner_act_np

            # accumulate per-env episode returns
            ep_return_running += rew
            ep_len_running += 1

            # episode boundary handling
            for i in range(N):
                if done_mask[i] > 0:
                    # log episode statistics
                    rolling_sparse.append(float(ep_return_running[i]))
                    rolling_len.append(int(ep_len_running[i]))
                    ep_return_running[i] = 0.0
                    ep_len_running[i] = 0
                    # replace partner, reset history
                    p, name, idx = train_pool.sample()
                    env.set_partner_for(i, p)
                    cur_opp_idx[i] = idx
                    cur_opp_name[i] = name
                    hist_obs[i] = 0.0
                    hist_act[i] = 0

            obs = new_obs

        # ---- bootstrap last value ----
        with torch.no_grad():
            last_obs_t = _t(obs, device)
            last_hist_obs_t = _t(hist_obs, device)
            last_hist_act_t = _t(hist_act, device, dtype=torch.long)
            _, _, last_v, _ = learner.act(last_obs_t, last_hist_obs_t, last_hist_act_t)

        adv, ret = compute_gae(buf_rew, buf_val, buf_done, last_v,
                               gamma=cfg.gamma, lam=cfg.gae_lambda)

        # ---- flatten ----
        obs_b = buf_obs.reshape(T * N, obs_dim)
        act_b = buf_act.reshape(T * N)
        logp_old = buf_logp.reshape(T * N)
        val_b = buf_val.reshape(T * N)
        adv_b = adv.reshape(T * N)
        ret_b = ret.reshape(T * N)
        hist_obs_b = buf_hist_obs.reshape(T * N, K, obs_dim)
        hist_act_b = buf_hist_act.reshape(T * N, K)
        p_act_b = buf_p_act.reshape(T * N)
        opp_idx_b = buf_opp_idx.reshape(T * N)
        adv_b = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)

        # ---- PPO + aux update ----
        n = obs_b.shape[0]
        mb_size = max(1, n // cfg.mini_batches)
        idx_all = np.arange(n)
        losses_p, losses_v, losses_e, losses_om, losses_c, kls = [], [], [], [], [], []
        accs_om = []
        for _ep in range(cfg.learning_epochs):
            np.random.shuffle(idx_all)
            for s in range(0, n, mb_size):
                mi = idx_all[s : s + mb_size]
                mi_t = torch.as_tensor(mi, device=device, dtype=torch.long)
                logp_new, ent, v_new, om_logits, _z = learner.evaluate(
                    obs_b[mi_t], hist_obs_b[mi_t], hist_act_b[mi_t], act_b[mi_t]
                )
                ratio = torch.exp(logp_new - logp_old[mi_t])
                s1 = ratio * adv_b[mi_t]
                s2 = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv_b[mi_t]
                p_loss = -torch.min(s1, s2).mean()
                v_clip = val_b[mi_t] + torch.clamp(v_new - val_b[mi_t], -cfg.clip_eps, cfg.clip_eps)
                v_loss = 0.5 * torch.max((v_new - ret_b[mi_t]) ** 2, (v_clip - ret_b[mi_t]) ** 2).mean()
                ent_loss = -ent.mean()

                if om_logits is not None:
                    om_loss = F.cross_entropy(om_logits, p_act_b[mi_t])
                    with torch.no_grad():
                        om_acc = (om_logits.argmax(-1) == p_act_b[mi_t]).float().mean()
                else:
                    om_loss = torch.zeros((), device=device)
                    om_acc = torch.zeros((), device=device)

                # ---- in-batch contrastive on the *trajectory* slice ----
                # Query: random chunk from each minibatch row
                # Positive: another chunk (here the same window the policy used)
                # in_batch_contrastive uses opp_idx as same/different mask
                if cfg.contrastive_coef > 0:
                    # We use hist windows we already encoded. For a second view,
                    # we encode a randomly-shifted slice from the same row.
                    shift = torch.randint(1, max(K // 4, 2), (mi_t.size(0),), device=device)
                    # construct shifted hist by rolling backward by `shift`
                    shifted_obs = torch.zeros_like(hist_obs_b[mi_t])
                    shifted_act = torch.zeros_like(hist_act_b[mi_t])
                    for j in range(mi_t.size(0)):
                        sj = int(shift[j].item())
                        shifted_obs[j, sj:] = hist_obs_b[mi_t[j], :K - sj]
                        shifted_act[j, sj:] = hist_act_b[mi_t[j], :K - sj]
                    z_q = learner.traj(hist_obs_b[mi_t], hist_act_b[mi_t])
                    with torch.no_grad():
                        z_k = traj_mom.encode_key(shifted_obs, shifted_act)
                    c_loss = in_batch_contrastive(z_q, z_k, opp_idx_b[mi_t], cfg.contrastive_temperature)
                else:
                    c_loss = torch.zeros((), device=device)

                loss = (
                    p_loss
                    + cfg.vf_coef * v_loss
                    + cfg.ent_coef * ent_loss
                    + cfg.om_coef * om_loss
                    + cfg.contrastive_coef * c_loss
                )
                opt.zero_grad(set_to_none=True)
                loss.backward()
                # clip all trainable params
                trainable = [p for g in optim_groups for p in g["params"]]
                nn.utils.clip_grad_norm_(trainable, cfg.max_grad_norm)
                opt.step()
                traj_mom.update_key()

                with torch.no_grad():
                    kls.append((logp_old[mi_t] - logp_new).mean().item())
                losses_p.append(p_loss.item())
                losses_v.append(v_loss.item())
                losses_e.append(ent_loss.item())
                losses_om.append(om_loss.item())
                losses_c.append(c_loss.item())
                accs_om.append(om_acc.item())

        # ---- log ----
        elapsed = time.time() - t0
        fps = int(global_step / max(elapsed, 1e-6))
        writer.add_scalar("perf/fps", fps, global_step)
        writer.add_scalar("loss/pi", float(np.mean(losses_p)), global_step)
        writer.add_scalar("loss/v", float(np.mean(losses_v)), global_step)
        writer.add_scalar("loss/ent", float(np.mean(losses_e)), global_step)
        writer.add_scalar("loss/om", float(np.mean(losses_om)), global_step)
        writer.add_scalar("loss/contrast", float(np.mean(losses_c)), global_step)
        writer.add_scalar("kl", float(np.mean(kls)), global_step)
        writer.add_scalar("om/acc", float(np.mean(accs_om)), global_step)
        writer.add_scalar("schedule/shaped_coef", shaped_coef, global_step)
        score_str = ""
        if rolling_sparse:
            mr = float(np.mean(rolling_sparse[-50:]))
            ml = float(np.mean(rolling_len[-50:]))
            writer.add_scalar("ep/return", mr, global_step)
            writer.add_scalar("ep/length", ml, global_step)
            score_str = f"ret={mr:6.2f}  len={ml:4.1f}"
        print(
            f"[{global_step:>9d}/{cfg.total_steps}] fps={fps:>5d}  {score_str}  "
            f"om_acc={np.mean(accs_om):.2f}  c_loss={np.mean(losses_c):.3f}  shaped_c={shaped_coef:.2f}"
        )

        # ---- ckpt ----
        if (global_step // cfg.ckpt_interval_steps) > (
            (global_step - T * N) // cfg.ckpt_interval_steps
        ):
            cpath = os.path.join(cfg.log_dir, f"ckpt_{global_step:09d}.pt")
            torch.save(
                {
                    "step": global_step,
                    "learner": learner.state_dict(),
                    "cfg": cfg.__dict__,
                    "obs_dim": obs_dim,
                    "n_actions": n_actions,
                    "pool_names": train_pool.names,
                },
                cpath,
            )
            print(f"  → saved {cpath}")

        if global_step >= cfg.total_steps:
            break

    final = os.path.join(cfg.log_dir, "ckpt_final.pt")
    torch.save(
        {
            "step": global_step,
            "learner": learner.state_dict(),
            "cfg": cfg.__dict__,
            "obs_dim": obs_dim,
            "n_actions": n_actions,
            "pool_names": train_pool.names,
        },
        final,
    )
    writer.close()
    env.close()
    print(f"done. final ckpt: {final}")
    return final
