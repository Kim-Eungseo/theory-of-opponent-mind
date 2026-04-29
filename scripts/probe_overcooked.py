"""Linear probing of Overcooked policy representations.

Question: how much partner-action information is in a *vanilla* policy's
encoder hidden state? We freeze the trained network, collect (h_t,
partner_action_t) pairs over many env steps, and train a tiny linear probe
on top of h_t to predict partner_action_t. The probe accuracy answers:
"is the partner info already implicit in the encoder, or do we have to
force it in?"

Compares two checkpoints:
* a vanilla baseline (no OM aux during training)
* a BAD-OM model (had explicit OM aux)

If vanilla probe ≫ chance, the intuition that "policy representation
implicitly contains partner info" is validated → the bottleneck of OM aux
is *utility*, not representation. If vanilla probe ≈ chance, the intuition
is wrong → we need to force partner info via aux/regularization.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tom.envs import VecOvercookedEnv
from tom.training.ippo_overcooked import ActorCriticOM, AGENT_IDS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_net(ckpt_path: str, env, om_in_policy: bool):
    nets = {a: ActorCriticOM(env.obs_dim, env.n_actions, hidden=256, om_in_policy=om_in_policy).to(DEVICE) for a in env.possible_agents}
    ck = torch.load(ckpt_path, map_location=DEVICE)
    for a in env.possible_agents:
        nets[a].load_state_dict(ck["nets"][a], strict=False)
        nets[a].eval()
    return nets


@torch.no_grad()
def collect_data(nets, env, n_steps: int = 20_000, agent: str = "agent_0"):
    """Run rollouts and collect (h_t, partner_action_t) for the given agent."""
    partner = AGENT_IDS[1] if agent == AGENT_IDS[0] else AGENT_IDS[0]
    obs = env.reset()

    h_buf = []
    pact_buf = []

    for _ in range(n_steps // env.num_envs):
        actions_np = {}
        my_h = None
        for a in env.possible_agents:
            o = torch.as_tensor(obs[a], dtype=torch.float32, device=DEVICE)
            if a == agent:
                my_h = nets[a].encoder(o)  # encoder hidden — what the probe sees
            act, _, _ = nets[a].act(o)
            actions_np[a] = act.cpu().numpy()

        h_buf.append(my_h.cpu().numpy())
        pact_buf.append(actions_np[partner].astype(np.int64))
        obs, _, _, _, _ = env.step(actions_np)

    H = np.concatenate(h_buf, axis=0)
    P = np.concatenate(pact_buf, axis=0)
    return H, P


def train_probe(H: np.ndarray, P: np.ndarray, n_classes: int, epochs: int = 30, lr: float = 1e-3, val_frac: float = 0.2):
    """Train a single Linear layer to predict P from H."""
    n = H.shape[0]
    idx = np.random.permutation(n)
    H = H[idx]; P = P[idx]
    n_val = int(n * val_frac)
    H_tr, H_va = H[n_val:], H[:n_val]
    P_tr, P_va = P[n_val:], P[:n_val]

    X_tr = torch.from_numpy(H_tr).float().to(DEVICE)
    Y_tr = torch.from_numpy(P_tr).long().to(DEVICE)
    X_va = torch.from_numpy(H_va).float().to(DEVICE)
    Y_va = torch.from_numpy(P_va).long().to(DEVICE)

    probe = nn.Linear(H.shape[1], n_classes).to(DEVICE)
    opt = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=1e-4)
    bs = 1024
    best_va = 0.0
    for ep in range(epochs):
        probe.train()
        for i in range(0, X_tr.shape[0], bs):
            xb = X_tr[i:i+bs]; yb = Y_tr[i:i+bs]
            logits = probe(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        probe.eval()
        with torch.no_grad():
            tr_acc = (probe(X_tr).argmax(-1) == Y_tr).float().mean().item()
            va_acc = (probe(X_va).argmax(-1) == Y_va).float().mean().item()
        best_va = max(best_va, va_acc)
    return tr_acc, va_acc, best_va


def majority_class_acc(P: np.ndarray, n_classes: int) -> float:
    counts = np.bincount(P, minlength=n_classes)
    return counts.max() / max(counts.sum(), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Checkpoint .pt path")
    ap.add_argument("--om-in-policy", action="store_true", help="Set if checkpoint was trained with --om-in-policy")
    ap.add_argument("--layout", default="asymmetric_advantages")
    ap.add_argument("--n-steps", type=int, default=40_000)
    ap.add_argument("--num-envs", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = VecOvercookedEnv(num_envs=args.num_envs, layout=args.layout, horizon=400, seed=args.seed)
    nets = load_net(args.ckpt, env, om_in_policy=args.om_in_policy)

    print(f"layout={args.layout}  ckpt={args.ckpt}  om_in_policy={args.om_in_policy}")
    print(f"obs_dim={env.obs_dim}  n_actions={env.n_actions}")

    rows = []
    for agent in env.possible_agents:
        H, P = collect_data(nets, env, n_steps=args.n_steps, agent=agent)
        chance = 1.0 / env.n_actions
        majority = majority_class_acc(P, env.n_actions)
        tr, va, best = train_probe(H, P, env.n_actions, epochs=args.epochs)
        print(f"  [{agent}]  N={H.shape[0]:>6}  chance={chance:.3f}  majority={majority:.3f}  "
              f"probe train={tr:.3f}  val={va:.3f}  best_val={best:.3f}")
        rows.append((agent, chance, majority, tr, va, best))

    print("\n--- summary ---")
    avg_best = np.mean([r[-1] for r in rows])
    print(f"avg probe best-val accuracy: {avg_best:.3f}")
    print(f"chance: {rows[0][1]:.3f}")
    print(f"avg majority-class baseline: {np.mean([r[2] for r in rows]):.3f}")


if __name__ == "__main__":
    main()
