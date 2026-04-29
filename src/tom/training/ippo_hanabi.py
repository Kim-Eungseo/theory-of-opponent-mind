"""Shared-policy PPO for 2-player Hanabi, with optional belief auxiliary head.

Architecture: a shared encoder feeds three heads — policy, value, and a
**belief** decoder that predicts the *acting player's own hand* (which is
hidden in the observation but available as ground truth from the env state).

Setting ``cfg.belief_coef = 0`` recovers the vanilla shared-policy IPPO
baseline (with the same architecture, so the comparison isolates the aux
loss). Setting it to a positive value adds the supervised belief loss to
the PPO objective.

Belief target: each card slot is a (color, rank) pair in {0..4}. Empty slots
(deck depleted near game end) are masked out of the loss.
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


# ---- Network -----------------------------------------------------------

def _orth_init(m: nn.Linear, gain: float = np.sqrt(2)):
    nn.init.orthogonal_(m.weight, gain=gain)
    nn.init.zeros_(m.bias)


class HanabiActorCritic(nn.Module):
    """Shared encoder + policy/value/belief heads.

    ``belief_in_policy=True`` enables BAD/SAD-style routing: the belief
    head's output (softmaxed per-card categoricals) is concatenated into
    the policy and value head inputs. This gives the policy *direct*
    access to the predicted hand belief — not just an implicit signal via
    the encoder. With this on, gradient from the policy loss also reaches
    the belief head, so the policy can shape belief output to be useful
    for action selection (alongside the supervised belief loss).
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden: int = 512,
        hand_size: int = 5,
        n_colors: int = 5,
        n_ranks: int = 5,
        belief_in_policy: bool = False,
    ):
        super().__init__()
        self.hand_size = hand_size
        self.n_colors = n_colors
        self.n_ranks = n_ranks
        self.belief_in_policy = bool(belief_in_policy)
        self._belief_feat_dim = hand_size * (n_colors + n_ranks)

        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        for m in self.encoder:
            if isinstance(m, nn.Linear):
                _orth_init(m, gain=np.sqrt(2))

        head_in = hidden + self._belief_feat_dim if self.belief_in_policy else hidden

        self.policy_head = nn.Linear(head_in, n_actions)
        _orth_init(self.policy_head, gain=0.01)

        self.value_head = nn.Linear(head_in, 1)
        _orth_init(self.value_head, gain=1.0)

        self.belief_head = nn.Linear(hidden, self._belief_feat_dim)
        _orth_init(self.belief_head, gain=0.1)

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs)

    def _split_belief(self, belief: torch.Tensor):
        belief_color = belief[:, : self.hand_size * self.n_colors].reshape(
            -1, self.hand_size, self.n_colors
        )
        belief_rank = belief[:, self.hand_size * self.n_colors :].reshape(
            -1, self.hand_size, self.n_ranks
        )
        return belief_color, belief_rank

    def _belief_feat(self, belief_color, belief_rank):
        # softmax over each card's color / rank for a clean probability feature
        bc = F.softmax(belief_color, dim=-1).flatten(1)
        br = F.softmax(belief_rank, dim=-1).flatten(1)
        return torch.cat([bc, br], dim=-1)

    def _heads(self, h: torch.Tensor, legal_mask: torch.Tensor):
        belief = self.belief_head(h)
        belief_color, belief_rank = self._split_belief(belief)
        if self.belief_in_policy:
            feat = torch.cat([h, self._belief_feat(belief_color, belief_rank)], dim=-1)
        else:
            feat = h
        logits = self.policy_head(feat).masked_fill(~legal_mask, -1e9)
        value = self.value_head(feat).squeeze(-1)
        return logits, value, belief_color, belief_rank

    def act(self, obs: torch.Tensor, legal_mask: torch.Tensor, deterministic: bool = False):
        h = self.encode(obs)
        logits, value, _, _ = self._heads(h, legal_mask)
        dist = torch.distributions.Categorical(logits=logits)
        a = logits.argmax(-1) if deterministic else dist.sample()
        logp = dist.log_prob(a)
        return a, logp, value

    def evaluate(
        self,
        obs: torch.Tensor,
        legal_mask: torch.Tensor,
        actions: torch.Tensor,
    ):
        h = self.encode(obs)
        logits, value, belief_color, belief_rank = self._heads(h, legal_mask)
        dist = torch.distributions.Categorical(logits=logits)
        logp = dist.log_prob(actions)
        ent = dist.entropy()
        return logp, ent, value, belief_color, belief_rank


# ---- GAE ---------------------------------------------------------------

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


# ---- Belief loss -------------------------------------------------------

def belief_loss_fn(
    belief_color: torch.Tensor,  # (B, hand_size, n_colors)
    belief_rank: torch.Tensor,   # (B, hand_size, n_ranks)
    target: torch.Tensor,        # (B, hand_size, 2) with -1 sentinel for empty
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (total loss, color accuracy, rank accuracy)."""
    target_color = target[:, :, 0]
    target_rank = target[:, :, 1]
    mask = target_color >= 0  # (B, hand_size)
    mask_f = mask.float()
    n = mask_f.sum().clamp(min=1.0)

    # safe targets (clamp -1 → 0; weight masks them out)
    tc = target_color.clamp(min=0).long()
    tr = target_rank.clamp(min=0).long()

    ce_color_per = F.cross_entropy(
        belief_color.reshape(-1, belief_color.shape[-1]),
        tc.reshape(-1),
        reduction="none",
    ).reshape(target_color.shape)
    ce_rank_per = F.cross_entropy(
        belief_rank.reshape(-1, belief_rank.shape[-1]),
        tr.reshape(-1),
        reduction="none",
    ).reshape(target_rank.shape)
    ce_color = (ce_color_per * mask_f).sum() / n
    ce_rank = (ce_rank_per * mask_f).sum() / n

    with torch.no_grad():
        pred_color = belief_color.argmax(-1)
        pred_rank = belief_rank.argmax(-1)
        acc_c = ((pred_color == tc).float() * mask_f).sum() / n
        acc_r = ((pred_rank == tr).float() * mask_f).sum() / n

    return ce_color + ce_rank, acc_c, acc_r


# ---- Config ------------------------------------------------------------

@dataclass
class HanabiPPOConfig:
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
    belief_coef: float = 0.0  # 0.0 disables belief aux; >0 enables
    belief_in_policy: bool = False  # BAD/SAD-style: feed belief output into policy/value heads
    max_grad_norm: float = 0.5
    hidden: int = 512

    log_dir: str = "runs_hanabi/ippo"
    log_interval: int = 1
    ckpt_interval_steps: int = 500_000
    seed: int = 0
    device: str = "auto"
    resume_from: str | None = None


def _t(x, device, dtype=torch.float32):
    return torch.as_tensor(x, dtype=dtype, device=device)


# ---- Trainer -----------------------------------------------------------

def train(cfg: HanabiPPOConfig) -> str:
    if cfg.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.device)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    env = VecHanabiEnv(num_envs=cfg.num_envs, env_name=cfg.env_name, num_players=NUM_PLAYERS, seed=cfg.seed)
    net = HanabiActorCritic(
        obs_dim=env.obs_dim,
        n_actions=env.n_actions,
        hidden=cfg.hidden,
        hand_size=env.hand_size,
        n_colors=env.n_colors,
        n_ranks=env.n_ranks,
        belief_in_policy=cfg.belief_in_policy,
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
        f"[cfg] belief_coef={cfg.belief_coef:.3f}  belief_in_policy={cfg.belief_in_policy}  "
        f"hidden={cfg.hidden}  num_envs={cfg.num_envs}  rollout={cfg.rollout_steps}  total={cfg.total_steps}"
    )

    obs, legal, cur, hands = env.reset()

    pending: list[list[dict | None]] = [[None] * NUM_PLAYERS for _ in range(cfg.num_envs)]
    accum: list[list[float]] = [[0.0] * NUM_PLAYERS for _ in range(cfg.num_envs)]
    trajs: list[list[list[dict]]] = [[[] for _ in range(NUM_PLAYERS)] for _ in range(cfg.num_envs)]

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
            with torch.no_grad():
                action_t, logp_t, value_t = net.act(obs_t, legal_t)
            actions = action_t.cpu().numpy().astype(np.int64)
            logprobs = logp_t.cpu().numpy().astype(np.float32)
            values = value_t.cpu().numpy().astype(np.float32)

            for i in range(cfg.num_envs):
                c = int(cur[i])
                if pending[i][c] is not None:
                    p = pending[i][c]
                    trajs[i][c].append(
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
                    "own_hand": hands[i, c].copy(),  # ground-truth own hand
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
                            trajs[i][p].append(
                                {
                                    "obs": pp["obs"], "action": pp["action"], "value": pp["value"],
                                    "logp": pp["logp"], "legal": pp["legal"],
                                    "own_hand": pp["own_hand"],
                                    "reward": float(accum[i][p]), "done": True,
                                }
                            )
                            pending[i][p] = None
                            accum[i][p] = 0.0
                    if info["completed"][i] is not None:
                        rolling_scores.append(info["completed"][i]["score"])
                        rolling_lengths.append(info["completed"][i]["length"])

            obs, legal, cur, hands = new_obs, new_legal, new_cur, new_hands

        # ----- bootstrap last value & build PPO batch -----
        all_obs, all_acts, all_legal = [], [], []
        all_logp_old, all_values, all_advs, all_returns = [], [], [], []
        all_own_hands = []

        for i in range(cfg.num_envs):
            for p in range(NUM_PLAYERS):
                traj = trajs[i][p]
                if not traj:
                    continue
                obs_arr = np.stack([s["obs"] for s in traj]).astype(np.float32)
                act_arr = np.array([s["action"] for s in traj], dtype=np.int64)
                lgl_arr = np.stack([s["legal"] for s in traj]).astype(bool)
                rew_arr = np.array([s["reward"] for s in traj], dtype=np.float32)
                done_arr = np.array([s["done"] for s in traj], dtype=np.float32)
                val_arr = np.array([s["value"] for s in traj], dtype=np.float32)
                logp_arr = np.array([s["logp"] for s in traj], dtype=np.float32)
                hand_arr = np.stack([s["own_hand"] for s in traj]).astype(np.int64)

                if traj[-1]["done"]:
                    last_v = 0.0
                elif pending[i][p] is not None:
                    last_v = float(pending[i][p]["value"])
                else:
                    last_v = 0.0

                rew_t = torch.from_numpy(rew_arr).to(device)
                val_t = torch.from_numpy(val_arr).to(device)
                done_t = torch.from_numpy(done_arr).to(device)
                advs, rets = gae_one_traj(
                    rew_t, val_t, done_t, torch.tensor(last_v, device=device),
                    gamma=cfg.gamma, lam=cfg.gae_lambda,
                )

                all_obs.append(torch.from_numpy(obs_arr).to(device))
                all_acts.append(torch.from_numpy(act_arr).to(device))
                all_legal.append(torch.from_numpy(lgl_arr).to(device))
                all_logp_old.append(torch.from_numpy(logp_arr).to(device))
                all_values.append(val_t)
                all_advs.append(advs)
                all_returns.append(rets)
                all_own_hands.append(torch.from_numpy(hand_arr).to(device))

        for i in range(cfg.num_envs):
            trajs[i] = [[] for _ in range(NUM_PLAYERS)]

        if not all_obs:
            continue

        obs_b = torch.cat(all_obs)
        act_b = torch.cat(all_acts)
        lgl_b = torch.cat(all_legal)
        logp_old = torch.cat(all_logp_old)
        val_b = torch.cat(all_values)
        adv_b = torch.cat(all_advs)
        ret_b = torch.cat(all_returns)
        hand_b = torch.cat(all_own_hands)
        adv_b = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)

        # ----- PPO update -----
        n = obs_b.shape[0]
        mb_size = max(1, n // cfg.mini_batches)
        idx = np.arange(n)
        losses_p, losses_v, losses_e, losses_b, kls = [], [], [], [], []
        accs_c, accs_r = [], []
        for _ep in range(cfg.learning_epochs):
            np.random.shuffle(idx)
            for s in range(0, n, mb_size):
                mi = torch.as_tensor(idx[s : s + mb_size], device=device, dtype=torch.long)
                logp_new, ent, v_new, b_color, b_rank = net.evaluate(obs_b[mi], lgl_b[mi], act_b[mi])
                ratio = torch.exp(logp_new - logp_old[mi])
                s1 = ratio * adv_b[mi]
                s2 = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv_b[mi]
                p_loss = -torch.min(s1, s2).mean()
                v_clip = val_b[mi] + torch.clamp(v_new - val_b[mi], -cfg.clip_eps, cfg.clip_eps)
                v_loss = 0.5 * torch.max((v_new - ret_b[mi]) ** 2, (v_clip - ret_b[mi]) ** 2).mean()
                ent_loss = -ent.mean()

                if cfg.belief_coef > 0:
                    b_loss, acc_c, acc_r = belief_loss_fn(b_color, b_rank, hand_b[mi])
                else:
                    b_loss = torch.zeros((), device=device)
                    with torch.no_grad():
                        _, acc_c, acc_r = belief_loss_fn(b_color, b_rank, hand_b[mi])

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
                    kls.append((logp_old[mi] - logp_new).mean().item())
                losses_p.append(p_loss.item())
                losses_v.append(v_loss.item())
                losses_e.append(ent_loss.item())
                losses_b.append(b_loss.item())
                accs_c.append(acc_c.item())
                accs_r.append(acc_r.item())

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
                f"belief_acc=(c={np.mean(accs_c):.2f},r={np.mean(accs_r):.2f})  n={n}"
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
