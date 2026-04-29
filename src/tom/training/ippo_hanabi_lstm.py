"""Recurrent (LSTM) shared-policy PPO for 2-player Hanabi.

Mirrors ``ippo_hanabi.py`` but replaces the MLP encoder with a 1-layer LSTM.
Each (env, player) keeps its own hidden state, reset at episode boundaries
— this matches the per-player MDP semantics already used in the MLP trainer.

Update: episodes are pushed to the PPO buffer only when fully *completed*
within a rollout. Each completed trajectory is replayed from ``h = 0`` so
gradients flow through the full LSTM rollout (true BPTT, episode-bounded).
Trajectories within a mini-batch are padded to the longest length and a
mask zeroes out the padding contribution to all losses.

Belief head behaviour is identical to the MLP version: ``belief_coef = 0``
disables the aux loss; positive values add it. The encoder gradient flows
from policy + value + belief; the policy / value / belief heads are
separate Linear layers, so policy gets no direct belief-loss gradient.
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

from tom.envs import VecHanabiEnv

NUM_PLAYERS = 2


def _orth_init(m: nn.Linear, gain: float = np.sqrt(2)):
    nn.init.orthogonal_(m.weight, gain=gain)
    nn.init.zeros_(m.bias)


# ---- Network -----------------------------------------------------------

class HanabiActorCriticLSTM(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden: int = 512,
        lstm_hidden: int = 512,
        hand_size: int = 5,
        n_colors: int = 5,
        n_ranks: int = 5,
    ):
        super().__init__()
        self.lstm_hidden = lstm_hidden
        self.hand_size = hand_size
        self.n_colors = n_colors
        self.n_ranks = n_ranks

        self.input_mlp = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
        )
        for m in self.input_mlp:
            if isinstance(m, nn.Linear):
                _orth_init(m, gain=np.sqrt(2))

        self.lstm = nn.LSTM(hidden, lstm_hidden, num_layers=1, batch_first=True)
        for name, p in self.lstm.named_parameters():
            if "weight_ih" in name or "weight_hh" in name:
                nn.init.orthogonal_(p)
            elif "bias" in name:
                nn.init.zeros_(p)

        self.policy_head = nn.Linear(lstm_hidden, n_actions)
        _orth_init(self.policy_head, gain=0.01)
        self.value_head = nn.Linear(lstm_hidden, 1)
        _orth_init(self.value_head, gain=1.0)
        self.belief_head = nn.Linear(lstm_hidden, hand_size * (n_colors + n_ranks))
        _orth_init(self.belief_head, gain=0.1)

    def init_hidden(self, batch_size: int, device) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.zeros(1, batch_size, self.lstm_hidden, device=device),
            torch.zeros(1, batch_size, self.lstm_hidden, device=device),
        )

    def step(
        self,
        obs: torch.Tensor,                       # (B, obs_dim)
        legal_mask: torch.Tensor,                # (B, n_actions)
        h: tuple[torch.Tensor, torch.Tensor],    # each (1, B, lstm_hidden)
        deterministic: bool = False,
    ):
        x = self.input_mlp(obs).unsqueeze(1)     # (B, 1, hidden)
        out, h_new = self.lstm(x, h)             # out (B, 1, lstm_hidden)
        feat = out.squeeze(1)
        logits = self.policy_head(feat).masked_fill(~legal_mask, -1e9)
        value = self.value_head(feat).squeeze(-1)
        dist = torch.distributions.Categorical(logits=logits)
        a = logits.argmax(-1) if deterministic else dist.sample()
        logp = dist.log_prob(a)
        return a, logp, value, h_new

    def forward_seq(
        self,
        obs_seq: torch.Tensor,                   # (B, T, obs_dim)
        legal_seq: torch.Tensor,                 # (B, T, n_actions)
        actions_seq: torch.Tensor,               # (B, T)
        h0: tuple[torch.Tensor, torch.Tensor] | None = None,
    ):
        B, T, _ = obs_seq.shape
        if h0 is None:
            h0 = self.init_hidden(B, obs_seq.device)
        x = self.input_mlp(obs_seq)              # (B, T, hidden)
        out, _ = self.lstm(x, h0)                # (B, T, lstm_hidden)
        logits = self.policy_head(out).masked_fill(~legal_seq, -1e9)
        values = self.value_head(out).squeeze(-1)
        belief = self.belief_head(out)           # (B, T, hand_size * (n_colors+n_ranks))
        belief_color = belief[..., : self.hand_size * self.n_colors].reshape(
            B, T, self.hand_size, self.n_colors
        )
        belief_rank = belief[..., self.hand_size * self.n_colors :].reshape(
            B, T, self.hand_size, self.n_ranks
        )
        dist = torch.distributions.Categorical(logits=logits)
        logp = dist.log_prob(actions_seq)        # (B, T)
        ent = dist.entropy()                     # (B, T)
        return logp, ent, values, belief_color, belief_rank


# ---- Belief loss (sequence-aware mask) ---------------------------------

def belief_loss_fn(
    belief_color: torch.Tensor,                  # (B, T, hand_size, n_colors)
    belief_rank: torch.Tensor,
    target: torch.Tensor,                        # (B, T, hand_size, 2)  -1 sentinel
    valid_mask: torch.Tensor,                    # (B, T) — sequence padding mask
):
    target_color = target[..., 0]                # (B, T, hand_size)
    target_rank = target[..., 1]
    card_mask = (target_color >= 0) & valid_mask.unsqueeze(-1)
    cm = card_mask.float()
    n = cm.sum().clamp(min=1.0)

    tc = target_color.clamp(min=0).long()
    tr = target_rank.clamp(min=0).long()

    ce_c_per = F.cross_entropy(
        belief_color.reshape(-1, belief_color.shape[-1]),
        tc.reshape(-1),
        reduction="none",
    ).reshape(target_color.shape)
    ce_r_per = F.cross_entropy(
        belief_rank.reshape(-1, belief_rank.shape[-1]),
        tr.reshape(-1),
        reduction="none",
    ).reshape(target_rank.shape)
    ce_c = (ce_c_per * cm).sum() / n
    ce_r = (ce_r_per * cm).sum() / n

    with torch.no_grad():
        pred_c = belief_color.argmax(-1)
        pred_r = belief_rank.argmax(-1)
        acc_c = ((pred_c == tc).float() * cm).sum() / n
        acc_r = ((pred_r == tr).float() * cm).sum() / n
    return ce_c + ce_r, acc_c, acc_r


def gae_one_traj(rewards, values, dones, last_value, gamma=0.99, lam=0.95):
    T = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    gae = 0.0
    for t in reversed(range(T)):
        next_v = last_value if t == T - 1 else values[t + 1]
        nonterm = 1.0 - dones[t].float()
        delta = rewards[t] + gamma * next_v * nonterm - values[t]
        gae = delta + gamma * lam * nonterm * gae
        advantages[t] = gae
    returns = advantages + values
    return advantages, returns


# ---- Config ------------------------------------------------------------

@dataclass
class HanabiLSTMPPOConfig:
    env_name: str = "Hanabi-Full-CardKnowledge"
    num_envs: int = 32
    total_steps: int = 5_000_000
    rollout_steps: int = 128
    learning_epochs: int = 4
    mini_batches: int = 4
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    belief_coef: float = 0.0
    max_grad_norm: float = 0.5
    hidden: int = 512
    lstm_hidden: int = 512

    log_dir: str = "runs_hanabi/ippo_lstm"
    log_interval: int = 1
    ckpt_interval_steps: int = 500_000
    seed: int = 0
    device: str = "auto"
    resume_from: str | None = None


def _t(x, device, dtype=torch.float32):
    return torch.as_tensor(x, dtype=dtype, device=device)


# ---- Trainer -----------------------------------------------------------

def train(cfg: HanabiLSTMPPOConfig) -> str:
    if cfg.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.device)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    env = VecHanabiEnv(num_envs=cfg.num_envs, env_name=cfg.env_name, num_players=NUM_PLAYERS, seed=cfg.seed)
    net = HanabiActorCriticLSTM(
        obs_dim=env.obs_dim, n_actions=env.n_actions,
        hidden=cfg.hidden, lstm_hidden=cfg.lstm_hidden,
        hand_size=env.hand_size, n_colors=env.n_colors, n_ranks=env.n_ranks,
    ).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)

    if cfg.resume_from:
        ck = torch.load(cfg.resume_from, map_location=device)
        net.load_state_dict(ck["net"], strict=False)
        opt.load_state_dict(ck["opt"])
        start_step = int(ck.get("step", 0))
        print(f"[resume] from {cfg.resume_from} @ step {start_step}")
    else:
        start_step = 0

    os.makedirs(cfg.log_dir, exist_ok=True)
    writer = SummaryWriter(cfg.log_dir)
    print(
        f"[cfg] LSTM  belief_coef={cfg.belief_coef:.3f}  "
        f"hidden={cfg.hidden}/lstm={cfg.lstm_hidden}  "
        f"num_envs={cfg.num_envs}  rollout={cfg.rollout_steps}  total={cfg.total_steps}"
    )

    obs, legal, cur, hands = env.reset()

    # per-(env, player) state
    pending = [[None] * NUM_PLAYERS for _ in range(cfg.num_envs)]
    accum = [[0.0] * NUM_PLAYERS for _ in range(cfg.num_envs)]
    # active episode buffer: list of transitions in current player's open episode
    ep_buf: list[list[list[dict]]] = [[[] for _ in range(NUM_PLAYERS)] for _ in range(cfg.num_envs)]
    # per-(env, player) LSTM hidden state — reset on episode end
    h_state = torch.zeros(1, cfg.num_envs * NUM_PLAYERS, cfg.lstm_hidden, device=device)
    c_state = torch.zeros(1, cfg.num_envs * NUM_PLAYERS, cfg.lstm_hidden, device=device)

    def _slot(env_i: int, player: int) -> int:
        return env_i * NUM_PLAYERS + player

    completed_trajs: list[list[dict]] = []
    rolling_scores: list[float] = []
    rolling_lengths: list[int] = []
    global_step = start_step
    iterations = (cfg.total_steps - start_step) // (cfg.rollout_steps * cfg.num_envs) + 1
    t0 = time.time()

    for it in range(iterations):
        # ----- collect rollout -----
        for _ in range(cfg.rollout_steps):
            obs_t = _t(obs, device)
            legal_t = _t(legal, device, dtype=torch.bool)

            # gather per-env current player's hidden slot
            slots = torch.tensor(
                [_slot(i, int(cur[i])) for i in range(cfg.num_envs)],
                device=device, dtype=torch.long,
            )
            h_in = h_state[:, slots, :]
            c_in = c_state[:, slots, :]
            with torch.no_grad():
                action_t, logp_t, value_t, (h_new, c_new) = net.step(obs_t, legal_t, (h_in, c_in))
            # write back updated hidden states
            with torch.no_grad():
                h_state[:, slots, :] = h_new
                c_state[:, slots, :] = c_new

            actions = action_t.cpu().numpy().astype(np.int64)
            logprobs = logp_t.cpu().numpy().astype(np.float32)
            values = value_t.cpu().numpy().astype(np.float32)

            for i in range(cfg.num_envs):
                c = int(cur[i])
                if pending[i][c] is not None:
                    p = pending[i][c]
                    ep_buf[i][c].append(
                        {
                            "obs": p["obs"], "action": p["action"], "value": p["value"],
                            "logp": p["logp"], "legal": p["legal"], "own_hand": p["own_hand"],
                            "reward": float(accum[i][c]), "done": False,
                        }
                    )
                    accum[i][c] = 0.0
                pending[i][c] = {
                    "obs": obs[i].copy(), "action": int(actions[i]),
                    "value": float(values[i]), "logp": float(logprobs[i]),
                    "legal": legal[i].copy(),
                    "own_hand": hands[i, c].copy(),
                }

            new_obs, new_legal, new_cur, new_hands, rewards, dones, info = env.step(actions)
            global_step += cfg.num_envs

            for i in range(cfg.num_envs):
                for p in range(NUM_PLAYERS):
                    accum[i][p] += float(rewards[i])
                if dones[i]:
                    for p in range(NUM_PLAYERS):
                        if pending[i][p] is not None:
                            pp = pending[i][p]
                            ep_buf[i][p].append(
                                {
                                    "obs": pp["obs"], "action": pp["action"], "value": pp["value"],
                                    "logp": pp["logp"], "legal": pp["legal"], "own_hand": pp["own_hand"],
                                    "reward": float(accum[i][p]), "done": True,
                                }
                            )
                            pending[i][p] = None
                            accum[i][p] = 0.0
                        # harvest the completed trajectory
                        if ep_buf[i][p]:
                            completed_trajs.append(ep_buf[i][p])
                            ep_buf[i][p] = []
                        # reset hidden state for that slot
                        s = _slot(i, p)
                        with torch.no_grad():
                            h_state[:, s, :].zero_()
                            c_state[:, s, :].zero_()
                    if info["completed"][i] is not None:
                        rolling_scores.append(info["completed"][i]["score"])
                        rolling_lengths.append(info["completed"][i]["length"])

            obs, legal, cur, hands = new_obs, new_legal, new_cur, new_hands

        if not completed_trajs:
            continue

        # ----- build padded tensors -----
        max_T = max(len(t) for t in completed_trajs)
        B = len(completed_trajs)
        obs_dim = env.obs_dim
        n_actions = env.n_actions
        hand_size = env.hand_size

        obs_pad = torch.zeros(B, max_T, obs_dim, device=device)
        act_pad = torch.zeros(B, max_T, dtype=torch.long, device=device)
        legal_pad = torch.zeros(B, max_T, n_actions, dtype=torch.bool, device=device)
        rew_pad = torch.zeros(B, max_T, device=device)
        done_pad = torch.zeros(B, max_T, device=device)
        val_pad = torch.zeros(B, max_T, device=device)
        logp_pad = torch.zeros(B, max_T, device=device)
        hand_pad = torch.full((B, max_T, hand_size, 2), -1, dtype=torch.long, device=device)
        valid_pad = torch.zeros(B, max_T, device=device)

        for b, traj in enumerate(completed_trajs):
            T = len(traj)
            obs_pad[b, :T] = torch.from_numpy(np.stack([s["obs"] for s in traj])).to(device)
            act_pad[b, :T] = torch.from_numpy(np.array([s["action"] for s in traj], dtype=np.int64)).to(device)
            legal_pad[b, :T] = torch.from_numpy(np.stack([s["legal"] for s in traj])).to(device)
            rew_pad[b, :T] = torch.from_numpy(np.array([s["reward"] for s in traj], dtype=np.float32)).to(device)
            done_pad[b, :T] = torch.from_numpy(np.array([s["done"] for s in traj], dtype=np.float32)).to(device)
            val_pad[b, :T] = torch.from_numpy(np.array([s["value"] for s in traj], dtype=np.float32)).to(device)
            logp_pad[b, :T] = torch.from_numpy(np.array([s["logp"] for s in traj], dtype=np.float32)).to(device)
            hand_pad[b, :T] = torch.from_numpy(np.stack([s["own_hand"] for s in traj])).to(device)
            valid_pad[b, :T] = 1.0
            # for padded positions, set legal_mask to all True so masked_fill doesn't produce NaNs
            if T < max_T:
                legal_pad[b, T:] = True

        # ----- per-trajectory GAE (each ends with done=True so bootstrap=0) -----
        adv_pad = torch.zeros_like(rew_pad)
        ret_pad = torch.zeros_like(rew_pad)
        for b in range(B):
            T = int(valid_pad[b].sum().item())
            if T == 0:
                continue
            advs, rets = gae_one_traj(
                rew_pad[b, :T], val_pad[b, :T], done_pad[b, :T],
                torch.tensor(0.0, device=device),
                gamma=cfg.gamma, lam=cfg.gae_lambda,
            )
            adv_pad[b, :T] = advs
            ret_pad[b, :T] = rets

        valid_total = valid_pad.sum().clamp(min=1.0)
        adv_mean = (adv_pad * valid_pad).sum() / valid_total
        adv_var = ((adv_pad - adv_mean) ** 2 * valid_pad).sum() / valid_total
        adv_pad = (adv_pad - adv_mean) / (adv_var.sqrt() + 1e-8)

        # ----- PPO update over completed trajectories -----
        idx = np.arange(B)
        mb_size = max(1, B // cfg.mini_batches)
        losses_p, losses_v, losses_e, losses_b, kls = [], [], [], [], []
        accs_c, accs_r = [], []
        for _ep in range(cfg.learning_epochs):
            np.random.shuffle(idx)
            for s in range(0, B, mb_size):
                mi = idx[s : s + mb_size]
                mi_t = torch.as_tensor(mi, device=device, dtype=torch.long)
                logp_new, ent, v_new, b_color, b_rank = net.forward_seq(
                    obs_pad[mi_t], legal_pad[mi_t], act_pad[mi_t]
                )

                vmask = valid_pad[mi_t]                  # (b, T)
                vmask_total = vmask.sum().clamp(min=1.0)

                ratio = torch.exp(logp_new - logp_pad[mi_t])
                s1 = ratio * adv_pad[mi_t]
                s2 = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv_pad[mi_t]
                p_loss_per = -torch.min(s1, s2)
                p_loss = (p_loss_per * vmask).sum() / vmask_total

                v_clip = val_pad[mi_t] + torch.clamp(v_new - val_pad[mi_t], -cfg.clip_eps, cfg.clip_eps)
                v_loss_per = 0.5 * torch.max((v_new - ret_pad[mi_t]) ** 2, (v_clip - ret_pad[mi_t]) ** 2)
                v_loss = (v_loss_per * vmask).sum() / vmask_total

                ent_loss = -(ent * vmask).sum() / vmask_total

                if cfg.belief_coef > 0:
                    b_loss, acc_c, acc_r = belief_loss_fn(b_color, b_rank, hand_pad[mi_t], vmask.bool())
                else:
                    b_loss = torch.zeros((), device=device)
                    with torch.no_grad():
                        _, acc_c, acc_r = belief_loss_fn(b_color, b_rank, hand_pad[mi_t], vmask.bool())

                loss = (
                    p_loss
                    + cfg.vf_coef * v_loss
                    + cfg.ent_coef * ent_loss
                    + cfg.belief_coef * b_loss
                )

                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
                opt.step()

                with torch.no_grad():
                    kls.append(((logp_pad[mi_t] - logp_new) * vmask).sum().item() / vmask_total.item())
                losses_p.append(p_loss.item())
                losses_v.append(v_loss.item())
                losses_e.append(ent_loss.item())
                losses_b.append(b_loss.item())
                accs_c.append(acc_c.item())
                accs_r.append(acc_r.item())

        # clear completed trajectories
        n_trans = sum(int(t["valid"]) if isinstance(t, dict) else len(t) for t in completed_trajs)
        completed_trajs = []

        # ----- logging -----
        elapsed = time.time() - t0
        fps = int(global_step / max(elapsed, 1e-6))
        if it % cfg.log_interval == 0:
            writer.add_scalar("perf/fps", fps, global_step)
            writer.add_scalar("loss/pi", float(np.mean(losses_p)), global_step)
            writer.add_scalar("loss/v", float(np.mean(losses_v)), global_step)
            writer.add_scalar("loss/ent", float(np.mean(losses_e)), global_step)
            writer.add_scalar("loss/belief", float(np.mean(losses_b)), global_step)
            writer.add_scalar("loss/kl", float(np.mean(kls)), global_step)
            writer.add_scalar("belief/color_acc", float(np.mean(accs_c)), global_step)
            writer.add_scalar("belief/rank_acc", float(np.mean(accs_r)), global_step)
            score_str = ""
            if rolling_scores:
                ms = float(np.mean(rolling_scores[-100:]))
                ml = float(np.mean(rolling_lengths[-100:]))
                writer.add_scalar("ep/score", ms, global_step)
                writer.add_scalar("ep/length", ml, global_step)
                score_str = f"score={ms:5.2f}  len={ml:4.1f}  "
            print(
                f"[{global_step:>9d}/{cfg.total_steps}] fps={fps:>6d}  {score_str}"
                f"belief_acc=(c={np.mean(accs_c):.2f},r={np.mean(accs_r):.2f})  trajs={B}"
            )

        # ----- ckpt -----
        if (global_step // cfg.ckpt_interval_steps) > (
            (global_step - cfg.rollout_steps * cfg.num_envs) // cfg.ckpt_interval_steps
        ):
            cpath = os.path.join(cfg.log_dir, f"ckpt_{global_step:09d}.pt")
            torch.save(
                {"step": global_step, "net": net.state_dict(), "opt": opt.state_dict(), "cfg": cfg.__dict__},
                cpath,
            )
            print(f"  → saved {cpath}")

        if global_step >= cfg.total_steps:
            break

    final = os.path.join(cfg.log_dir, "ckpt_final.pt")
    torch.save({"step": global_step, "net": net.state_dict(), "opt": opt.state_dict(), "cfg": cfg.__dict__}, final)
    writer.close()
    env.close()
    print(f"done. final ckpt: {final}")
    return final
