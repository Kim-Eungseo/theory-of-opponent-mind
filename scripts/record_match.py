"""Record a match as mp4. Side-by-side: [player_0 view | player_1 view].

Usage:
    python scripts/record_match.py --out match.mp4 --seconds 10
    python scripts/record_match.py --ckpt path/to/policy.pt --out trained.mp4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tom.envs import VizDoomMultiAgentEnv


def _load_policy(ckpt: str, n_actions: int, n_vars: int, device: str = "cpu"):
    import torch

    from tom.training.ppo import CNNActorCritic

    net = CNNActorCritic(n_actions, n_vars).to(device)
    state = torch.load(ckpt, map_location=device, weights_only=True)
    net.load_state_dict(state)
    net.eval()
    return net


def _act_policy(net, obs: dict, agents: list[str], device: str = "cpu"):
    import torch

    screen = np.stack([obs[a]["screen"] for a in agents])
    gv = np.stack([obs[a]["gamevars"] for a in agents])
    with torch.no_grad():
        logits, _ = net(
            torch.from_numpy(screen).to(device),
            torch.from_numpy(gv).to(device),
        )
        actions = torch.distributions.Categorical(logits=logits).sample()
    return {a: int(actions[i].item()) for i, a in enumerate(agents)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="match.mp4")
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fps", type=int, default=35)
    ap.add_argument("--ckpt", type=str, default=None,
                    help="PPO checkpoint .pt; if omitted, random policy")
    ap.add_argument("--ticrate", type=int, default=1000)
    ap.add_argument("--frame-skip", type=int, default=1,
                    help="1 + fps=35 for real game speed. default=1")
    ap.add_argument("--scenario", type=str, default=None,
                    help="cfg filename under vizdoom scenarios_path, or abs path")
    ap.add_argument("--num-players", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    scenario = args.scenario
    if scenario and not scenario.startswith("/") and not scenario.endswith(".cfg"):
        scenario = scenario + ".cfg"
    if scenario and not scenario.startswith("/"):
        import vizdoom as vzd
        scenario = str(Path(vzd.scenarios_path) / scenario)

    env = VizDoomMultiAgentEnv(
        num_players=args.num_players,
        scenario=scenario,
        episode_timeout_seconds=args.seconds,
        ticrate=args.ticrate,
        frame_skip=args.frame_skip,
        seed=args.seed,
        record=True,
    )

    policy_net = None
    device = "cpu"
    if args.ckpt:
        try:
            import torch
            device = args.device if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
        policy_net = _load_policy(
            args.ckpt, env.n_buttons, env.n_vars, device=device
        )
        print(f"loaded policy from {args.ckpt} (device={device})")
    else:
        print("using random policy")

    # target frame count: one per tic of target game-time
    target_frames = int(args.seconds * 35 / max(args.frame_skip, 1))

    obs, _ = env.reset()
    frames: list[np.ndarray] = []
    returns = {a: 0.0 for a in env.possible_agents}
    episodes = 0

    while len(frames) < target_frames:
        if not env.agents:
            # env ended a round — auto-reset for a continuous video
            obs, _ = env.reset()
            episodes += 1

        if policy_net is None:
            actions = {a: env.action_space(a).sample() for a in env.agents}
        else:
            actions = _act_policy(policy_net, obs, list(env.agents), device=device)

        obs, rewards, terms, truncs, infos = env.step(actions)
        for a, r in rewards.items():
            returns[a] += r

        views = [infos[a]["frame"] for a in sorted(infos)]
        labeled = []
        for i, v in enumerate(views):
            v2 = v.copy()
            cv2.putText(
                v2, f"P{i}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 0), 2, cv2.LINE_AA,
            )
            labeled.append(v2)
        frames.append(np.hstack(labeled))

    env.close()
    print(f"captured {len(frames)} frames across {episodes + 1} episode(s)")

    if not frames:
        print("no frames captured")
        return

    H, W = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        args.out,
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (W, H),
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not open VideoWriter for {args.out}")
    for frame in frames:
        # cv2 writes BGR, our frames are RGB
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    print(
        f"saved {len(frames)} frames ({W}x{H} @ {args.fps}fps) -> {args.out} | "
        f"returns={ {k: round(v,2) for k,v in returns.items()} }"
    )


if __name__ == "__main__":
    main()
