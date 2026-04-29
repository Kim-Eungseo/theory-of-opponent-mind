"""IPPO trainer for the multi-agent Overcooked env, with optional OM aux head.

Each agent has its own actor-critic. ``om_coef`` adds a supervised aux loss
that predicts the *other* agent's action at the same timestep — the ground
truth is just the partner's actual action recorded during rollout.

When ``om_in_policy=True``, the OM head's output (softmax over partner
actions) is concatenated into the policy/value head input — BAD/SAD-style
routing, so the policy can use the predicted partner action directly. The
gradient from the policy loss then also reaches the OM head, alongside the
supervised cross-entropy loss.

Architecture: shared 2-layer encoder, three heads (policy / value / OM).
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from tom.envs import VecOvercookedEnv

AGENT_IDS = ("agent_0", "agent_1")


# ---- Network -----------------------------------------------------------

def _orth(m: nn.Linear, gain=np.sqrt(2)):
    nn.init.orthogonal_(m.weight, gain=gain)
    nn.init.zeros_(m.bias)


class TrajectoryOM(nn.Module):
    """Predict partner's *current* action from their last K observations.

    Small recurrent partner-model: LSTM over a sliding obs window. Distinct
    from the cur-step OM head and from SOM (policy-as-OM) — has its own
    weights. Trained as a supervised aux task; optionally its softmax
    output can be routed into the policy/value heads (BAD-style).
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.lstm = nn.LSTM(obs_dim, hidden, batch_first=True)
        for name, p in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(p)
            elif "bias" in name:
                nn.init.zeros_(p)
        self.head = nn.Linear(hidden, n_actions)
        _orth(self.head, gain=0.1)

    def forward(self, hist: torch.Tensor) -> torch.Tensor:
        # hist: (B, K, obs_dim)
        out, _ = self.lstm(hist)
        return self.head(out[:, -1, :])  # (B, n_actions)


class ActorCriticOM(nn.Module):
    """Shared encoder + policy / value / OM heads.

    OM head predicts the partner's simultaneous action distribution.
    Optional TrajectoryOM (TOM) sub-module handles partner trajectory
    modelling with its own LSTM.
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden: int = 256,
        om_in_policy: bool = False,
        use_tom: bool = False,
        tom_hidden: int = 128,
        tom_in_policy: bool = False,
    ):
        super().__init__()
        self.n_actions = n_actions
        self.om_in_policy = bool(om_in_policy)
        self.use_tom = bool(use_tom)
        self.tom_in_policy = bool(tom_in_policy) and self.use_tom

        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        for m in self.encoder:
            if isinstance(m, nn.Linear):
                _orth(m, gain=np.sqrt(2))

        self.om_head = nn.Linear(hidden, n_actions)
        _orth(self.om_head, gain=0.1)

        if self.use_tom:
            self.tom_head = TrajectoryOM(obs_dim, n_actions, hidden=tom_hidden)
        else:
            self.tom_head = None

        head_in = hidden
        if self.om_in_policy:
            head_in += n_actions
        if self.tom_in_policy:
            head_in += n_actions
        self.policy_head = nn.Linear(head_in, n_actions)
        _orth(self.policy_head, gain=0.01)
        self.value_head = nn.Linear(head_in, 1)
        _orth(self.value_head, gain=1.0)

    def _heads(self, h: torch.Tensor, tom_logits: torch.Tensor | None = None):
        om_logits = self.om_head(h)
        feats = [h]
        if self.om_in_policy:
            feats.append(F.softmax(om_logits, dim=-1))
        if self.tom_in_policy:
            assert tom_logits is not None
            feats.append(F.softmax(tom_logits, dim=-1))
        feat = torch.cat(feats, dim=-1) if len(feats) > 1 else h
        logits = self.policy_head(feat)
        value = self.value_head(feat).squeeze(-1)
        return logits, value, om_logits

    def act(self, obs: torch.Tensor, deterministic: bool = False, partner_hist: torch.Tensor | None = None):
        h = self.encoder(obs)
        tom_logits = None
        if self.tom_in_policy:
            assert partner_hist is not None
            tom_logits = self.tom_head(partner_hist)
        logits, value, _ = self._heads(h, tom_logits)
        dist = torch.distributions.Categorical(logits=logits)
        a = logits.argmax(-1) if deterministic else dist.sample()
        logp = dist.log_prob(a)
        return a, logp, value

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor, partner_hist: torch.Tensor | None = None):
        h = self.encoder(obs)
        tom_logits_for_policy = None
        if self.tom_in_policy:
            assert partner_hist is not None
            tom_logits_for_policy = self.tom_head(partner_hist)
        logits, value, om_logits = self._heads(h, tom_logits_for_policy)
        dist = torch.distributions.Categorical(logits=logits)
        logp = dist.log_prob(actions)
        ent = dist.entropy()
        # also compute TOM aux output (independent of routing) so the trainer can supervise it
        tom_aux_logits = None
        if self.use_tom and partner_hist is not None:
            tom_aux_logits = (
                tom_logits_for_policy if tom_logits_for_policy is not None else self.tom_head(partner_hist)
            )
        return logp, ent, value, om_logits, tom_aux_logits

    def value_only(self, obs: torch.Tensor, partner_hist: torch.Tensor | None = None):
        h = self.encoder(obs)
        tom_logits = None
        if self.tom_in_policy:
            assert partner_hist is not None
            tom_logits = self.tom_head(partner_hist)
        _, value, _ = self._heads(h, tom_logits)
        return value


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


# ---- Config ------------------------------------------------------------

@dataclass
class IPPOConfig:
    layout: str = "cramped_room"
    num_envs: int = 16
    horizon: int = 400
    total_steps: int = 1_000_000
    rollout: int = 400
    learning_epochs: int = 8
    mini_batches: int = 4
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.98
    clip_eps: float = 0.05
    ent_coef: float = 0.1
    vf_coef: float = 0.5
    om_coef: float = 0.0
    om_in_policy: bool = False
    som_coef: float = 0.0  # Self-Other Modeling: pass partner obs through my policy, NLL-fit partner action
    tom_coef: float = 0.0  # Trajectory OM: separate LSTM head over partner's last K obs
    tom_history_len: int = 8
    tom_hidden: int = 128
    tom_in_policy: bool = False  # BAD-style: route TOM softmax into policy/value head input
    max_grad_norm: float = 0.5
    hidden: int = 256

    shaped_reward_coef_start: float = 1.0
    shaped_reward_coef_end: float = 0.0
    shaped_reward_anneal_frac: float = 0.5

    log_dir: str = "runs_overcooked/ippo"
    log_interval: int = 1
    ckpt_interval_steps: int = 200_000
    seed: int = 0
    device: str = "auto"
    resume_from: str | None = None


def _tensor(x, device):
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def train(cfg: IPPOConfig) -> str:
    if cfg.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.device)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    env = VecOvercookedEnv(
        num_envs=cfg.num_envs,
        layout=cfg.layout,
        horizon=cfg.horizon,
        shaped_reward_coef=cfg.shaped_reward_coef_start,
        seed=cfg.seed,
    )

    use_tom = cfg.tom_coef > 0 or cfg.tom_in_policy
    nets = {
        a: ActorCriticOM(
            env.obs_dim, env.n_actions, cfg.hidden,
            om_in_policy=cfg.om_in_policy,
            use_tom=use_tom,
            tom_hidden=cfg.tom_hidden,
            tom_in_policy=cfg.tom_in_policy,
        ).to(device)
        for a in env.possible_agents
    }
    opts = {a: torch.optim.Adam(nets[a].parameters(), lr=cfg.lr) for a in env.possible_agents}

    print(
        f"[cfg] om_coef={cfg.om_coef:.3f}  om_in_policy={cfg.om_in_policy}  "
        f"som_coef={cfg.som_coef:.3f}  tom_coef={cfg.tom_coef:.3f}  "
        f"tom_K={cfg.tom_history_len}  tom_in_policy={cfg.tom_in_policy}  "
        f"layout={cfg.layout}  num_envs={cfg.num_envs}  total={cfg.total_steps}"
    )

    if cfg.resume_from:
        ck = torch.load(cfg.resume_from, map_location=device)
        for a in env.possible_agents:
            nets[a].load_state_dict(ck["nets"][a], strict=False)
            opts[a].load_state_dict(ck["opts"][a])
        start_step = int(ck.get("step", 0))
        print(f"[resume] loaded {cfg.resume_from} @ step {start_step}")
    else:
        start_step = 0

    os.makedirs(cfg.log_dir, exist_ok=True)
    writer = SummaryWriter(cfg.log_dir)

    obs = env.reset(seed=cfg.seed)

    global_step = start_step
    iterations = (cfg.total_steps - start_step) // (cfg.rollout * cfg.num_envs) + 1
    rolling_sparse = []
    rolling_shaped = []
    t0 = time.time()

    # partner-of mapping
    partner_of = {AGENT_IDS[0]: AGENT_IDS[1], AGENT_IDS[1]: AGENT_IDS[0]}

    for it in range(iterations):
        frac = max(0.0, min(1.0, global_step / max(1, int(cfg.total_steps * cfg.shaped_reward_anneal_frac))))
        shaped_coef = cfg.shaped_reward_coef_start + frac * (cfg.shaped_reward_coef_end - cfg.shaped_reward_coef_start)
        for sub in env.envs:
            sub.shaped_reward_coef = shaped_coef

        T, N = cfg.rollout, cfg.num_envs
        buf = {
            a: {
                "obs": torch.zeros((T, N, env.obs_dim), dtype=torch.float32, device=device),
                "partner_obs": torch.zeros((T, N, env.obs_dim), dtype=torch.float32, device=device),
                "actions": torch.zeros((T, N), dtype=torch.long, device=device),
                "partner_actions": torch.zeros((T, N), dtype=torch.long, device=device),
                "logprobs": torch.zeros((T, N), dtype=torch.float32, device=device),
                "values": torch.zeros((T, N), dtype=torch.float32, device=device),
                "rewards": torch.zeros((T, N), dtype=torch.float32, device=device),
                "dones": torch.zeros((T, N), dtype=torch.float32, device=device),
            }
            for a in env.possible_agents
        }

        # for tom_in_policy: maintain a sliding window of recent partner_obs per agent
        if cfg.tom_in_policy:
            phist = {
                a: torch.zeros(N, cfg.tom_history_len, env.obs_dim, dtype=torch.float32, device=device)
                for a in env.possible_agents
            }
        else:
            phist = None

        for t in range(T):
            # update partner-history buffer with current partner_obs (then act will see it)
            if cfg.tom_in_policy:
                for a in env.possible_agents:
                    pa = partner_of[a]
                    cur_pobs = _tensor(obs[pa], device)
                    phist[a] = torch.cat([phist[a][:, 1:, :], cur_pobs.unsqueeze(1)], dim=1)

            actions_np = {}
            with torch.no_grad():
                for a in env.possible_agents:
                    o = _tensor(obs[a], device)
                    act, logp, v = nets[a].act(o, partner_hist=phist[a] if cfg.tom_in_policy else None)
                    buf[a]["obs"][t] = o
                    buf[a]["actions"][t] = act
                    buf[a]["logprobs"][t] = logp
                    buf[a]["values"][t] = v
                    actions_np[a] = act.cpu().numpy()

            # write partner-obs and partner-action ground truth for each agent
            for a in env.possible_agents:
                pa = partner_of[a]
                buf[a]["partner_actions"][t] = buf[pa]["actions"][t]
                buf[a]["partner_obs"][t] = buf[pa]["obs"][t]

            obs, rew, _term, trunc, info = env.step(actions_np)

            done_mask = np.zeros(N, dtype=np.float32)
            for i in range(N):
                if info["completed"][i] is not None:
                    rolling_sparse.append(info["completed"][i]["sparse_return"])
                    rolling_shaped.append(info["completed"][i]["shaped_return"])
                    done_mask[i] = 1.0
                    # zero out partner history for that env on episode end
                    if cfg.tom_in_policy:
                        for a in env.possible_agents:
                            phist[a][i].zero_()
            for a in env.possible_agents:
                buf[a]["rewards"][t] = _tensor(rew[a], device)
                buf[a]["dones"][t] = _tensor(done_mask, device)

            global_step += N

        with torch.no_grad():
            last_values = {
                a: nets[a].value_only(
                    _tensor(obs[a], device),
                    partner_hist=phist[a] if cfg.tom_in_policy else None,
                )
                for a in env.possible_agents
            }

        for a in env.possible_agents:
            advs, rets = compute_gae(
                buf[a]["rewards"], buf[a]["values"], buf[a]["dones"], last_values[a],
                gamma=cfg.gamma, lam=cfg.gae_lambda,
            )
            buf[a]["advantages"] = advs
            buf[a]["returns"] = rets

        log_metrics = {}
        K = cfg.tom_history_len
        for a in env.possible_agents:
            d = buf[a]
            obs_b = d["obs"].reshape(T * N, -1)
            pobs_b = d["partner_obs"].reshape(T * N, -1)
            act_b = d["actions"].reshape(T * N)
            pact_b = d["partner_actions"].reshape(T * N)
            logp_old = d["logprobs"].reshape(T * N)
            adv_b = d["advantages"].reshape(T * N)
            ret_b = d["returns"].reshape(T * N)
            val_b = d["values"].reshape(T * N)
            adv_b = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)

            # build partner-history windows once: (T, N, K, obs_dim) → (T*N, K, obs_dim)
            if use_tom:
                partner_obs_full = d["partner_obs"]  # (T, N, obs_dim)
                pad = torch.zeros(K - 1, N, env.obs_dim, dtype=torch.float32, device=device)
                padded = torch.cat([pad, partner_obs_full], dim=0)  # (T+K-1, N, obs_dim)
                hist = padded.unfold(0, K, 1).permute(0, 1, 3, 2).contiguous()  # (T, N, K, obs_dim)
                hist_b = hist.reshape(T * N, K, env.obs_dim)
            else:
                hist_b = None

            n = obs_b.shape[0]
            mb_size = n // cfg.mini_batches
            idx = np.arange(n)
            losses_p, losses_v, losses_e, losses_om, losses_som, losses_tom = [], [], [], [], [], []
            kls, om_accs, som_accs, tom_accs = [], [], [], []
            for _ep in range(cfg.learning_epochs):
                np.random.shuffle(idx)
                for s in range(0, n, mb_size):
                    mi = idx[s : s + mb_size]
                    mi_t = torch.as_tensor(mi, device=device)
                    hist_mb = hist_b[mi_t] if use_tom else None
                    logp_new, ent, v_new, om_logits, tom_logits = nets[a].evaluate(
                        obs_b[mi_t], act_b[mi_t], partner_hist=hist_mb,
                    )
                    ratio = torch.exp(logp_new - logp_old[mi_t])
                    s1 = ratio * adv_b[mi_t]
                    s2 = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv_b[mi_t]
                    p_loss = -torch.min(s1, s2).mean()
                    v_clip = val_b[mi_t] + torch.clamp(v_new - val_b[mi_t], -cfg.clip_eps, cfg.clip_eps)
                    v_loss = 0.5 * torch.max((v_new - ret_b[mi_t]) ** 2, (v_clip - ret_b[mi_t]) ** 2).mean()
                    ent_loss = -ent.mean()

                    om_loss = F.cross_entropy(om_logits, pact_b[mi_t])
                    with torch.no_grad():
                        om_acc = (om_logits.argmax(-1) == pact_b[mi_t]).float().mean()

                    if cfg.som_coef > 0:
                        logp_som, _, _, _, _ = nets[a].evaluate(pobs_b[mi_t], pact_b[mi_t])
                        som_loss = -logp_som.mean()
                        with torch.no_grad():
                            h_som = nets[a].encoder(pobs_b[mi_t])
                            tom_logits_som = nets[a].tom_head(hist_mb) if (nets[a].tom_in_policy and hist_mb is not None) else None
                            logits_som, _, _ = nets[a]._heads(h_som, tom_logits_som)
                            som_acc = (logits_som.argmax(-1) == pact_b[mi_t]).float().mean()
                    else:
                        som_loss = torch.zeros((), device=device)
                        som_acc = torch.zeros((), device=device)

                    if cfg.tom_coef > 0 and tom_logits is not None:
                        tom_loss = F.cross_entropy(tom_logits, pact_b[mi_t])
                        with torch.no_grad():
                            tom_acc = (tom_logits.argmax(-1) == pact_b[mi_t]).float().mean()
                    else:
                        tom_loss = torch.zeros((), device=device)
                        tom_acc = torch.zeros((), device=device)

                    loss = (
                        p_loss
                        + cfg.vf_coef * v_loss
                        + cfg.ent_coef * ent_loss
                        + cfg.om_coef * om_loss
                        + cfg.som_coef * som_loss
                        + cfg.tom_coef * tom_loss
                    )

                    opts[a].zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(nets[a].parameters(), cfg.max_grad_norm)
                    opts[a].step()

                    with torch.no_grad():
                        kls.append((logp_old[mi_t] - logp_new).mean().item())
                    losses_p.append(p_loss.item())
                    losses_v.append(v_loss.item())
                    losses_e.append(ent_loss.item())
                    losses_om.append(om_loss.item())
                    losses_som.append(som_loss.item())
                    losses_tom.append(tom_loss.item())
                    om_accs.append(om_acc.item())
                    som_accs.append(som_acc.item())
                    tom_accs.append(tom_acc.item())

            log_metrics[a] = {
                "loss/pi": float(np.mean(losses_p)),
                "loss/v": float(np.mean(losses_v)),
                "loss/ent": float(np.mean(losses_e)),
                "loss/om": float(np.mean(losses_om)),
                "loss/som": float(np.mean(losses_som)),
                "loss/tom": float(np.mean(losses_tom)),
                "om/acc": float(np.mean(om_accs)),
                "som/acc": float(np.mean(som_accs)),
                "tom/acc": float(np.mean(tom_accs)),
                "kl": float(np.mean(kls)),
            }

        elapsed = time.time() - t0
        fps = int(global_step / max(elapsed, 1e-6))
        if it % cfg.log_interval == 0:
            writer.add_scalar("perf/fps", fps, global_step)
            writer.add_scalar("schedule/shaped_coef", shaped_coef, global_step)
            for a in env.possible_agents:
                for k, v in log_metrics[a].items():
                    writer.add_scalar(f"{a}/{k}", v, global_step)
            score_str = ""
            if rolling_sparse:
                ms = float(np.mean(rolling_sparse[-50:]))
                msh = float(np.mean(rolling_shaped[-50:]))
                writer.add_scalar("ep/sparse_return", ms, global_step)
                writer.add_scalar("ep/shaped_return", msh, global_step)
                avg_om_acc = float(np.mean([log_metrics[a]["om/acc"] for a in env.possible_agents]))
                avg_som_acc = float(np.mean([log_metrics[a]["som/acc"] for a in env.possible_agents]))
                avg_tom_acc = float(np.mean([log_metrics[a]["tom/acc"] for a in env.possible_agents]))
                writer.add_scalar("om/acc_mean", avg_om_acc, global_step)
                writer.add_scalar("som/acc_mean", avg_som_acc, global_step)
                writer.add_scalar("tom/acc_mean", avg_tom_acc, global_step)
                score_str = (
                    f"sparse={ms:6.2f}  shaped={msh:7.2f}  "
                    f"om={avg_om_acc:.2f}  som={avg_som_acc:.2f}  tom={avg_tom_acc:.2f}  "
                    f"shaped_coef={shaped_coef:.2f}"
                )
                print(f"[{global_step:>9d}/{cfg.total_steps}] fps={fps:>6d}  {score_str}")

        if (global_step // cfg.ckpt_interval_steps) > (
            (global_step - cfg.rollout * cfg.num_envs) // cfg.ckpt_interval_steps
        ):
            cpath = os.path.join(cfg.log_dir, f"ckpt_{global_step:09d}.pt")
            torch.save(
                {
                    "step": global_step,
                    "nets": {a: nets[a].state_dict() for a in env.possible_agents},
                    "opts": {a: opts[a].state_dict() for a in env.possible_agents},
                    "cfg": cfg.__dict__,
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
            "nets": {a: nets[a].state_dict() for a in env.possible_agents},
            "opts": {a: opts[a].state_dict() for a in env.possible_agents},
            "cfg": cfg.__dict__,
        },
        final,
    )
    writer.close()
    env.close()
    print(f"done. final ckpt: {final}")
    return final
