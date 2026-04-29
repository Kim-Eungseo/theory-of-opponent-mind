"""Self-play IPPO on the multi-agent ViZDoom env, via skrl.

Uses skrl 2.x Independent-PPO with an IMPALA-style ResNet backbone. The env
is vectorised through ``VecVizDoomMultiAgentEnv`` so multiple matches run in
parallel — the env-side bottleneck (ViZDoom simulation + pipe IPC) is the
dominant cost, and running 4~8 envs keeps the GPU fed.

Shared-parameter self-play is NOT used: each agent has its own policy /
value network (classic IPPO). Parameter sharing is a separate, later change.

Requires:
    pip install skrl tqdm
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from skrl.agents.torch import ExperimentCfg
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import CategoricalMixin, DeterministicMixin, Model
from skrl.multi_agents.torch.ippo import IPPO, IPPO_CFG
from skrl.trainers.torch import SequentialTrainer
from skrl.utils.spaces.torch import unflatten_tensorized_space

from tom.envs import VecVizDoomMultiAgentEnv, VizDoomMultiAgentEnv


# ---- IMPALA-style ResNet backbone --------------------------------------

class _ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        y = F.relu(x)
        y = self.conv1(y)
        y = F.relu(y)
        y = self.conv2(y)
        return x + y


class _ImpalaBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.res1 = _ResBlock(out_ch)
        self.res2 = _ResBlock(out_ch)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = self.res1(x)
        x = self.res2(x)
        return x


class ImpalaCNN(nn.Module):
    """IMPALA ResNet (Espeholt et al. 2018). ~4M params at widths (32, 64, 64)."""

    def __init__(self, widths: tuple[int, ...] = (32, 64, 64)):
        super().__init__()
        blocks = []
        prev = 3
        for w in widths:
            blocks.append(_ImpalaBlock(prev, w))
            prev = w
        self.body = nn.Sequential(*blocks)

    def forward(self, x):
        x = self.body(x)
        x = F.relu(x)
        return torch.flatten(x, 1)


def _make_backbone(n_vars: int, hidden: int = 512):
    cnn = ImpalaCNN(widths=(32, 64, 64))
    with torch.no_grad():
        feat_dim = cnn(torch.zeros(1, 3, 84, 84)).shape[1]
    trunk = nn.Sequential(
        nn.Linear(feat_dim + n_vars, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
    )
    return cnn, trunk, hidden


class DoomActor(CategoricalMixin, Model):
    def __init__(self, observation_space, action_space, device, hidden: int = 512):
        Model.__init__(
            self,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
        )
        CategoricalMixin.__init__(self, unnormalized_log_prob=True)
        n_vars = observation_space["gamevars"].shape[0]
        self.cnn, self.trunk, h = _make_backbone(n_vars, hidden)
        self.head = nn.Linear(h, action_space.n)

    def compute(self, inputs, role):
        obs = inputs.get("observations", inputs.get("states"))
        d = unflatten_tensorized_space(self.observation_space, obs)
        screen = d["screen"].float() / 255.0
        gv = d["gamevars"].float()
        feat = self.cnn(screen)
        h = self.trunk(torch.cat([feat, gv], dim=-1))
        return self.head(h), {}


class DoomCritic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, hidden: int = 512):
        Model.__init__(
            self,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
        )
        DeterministicMixin.__init__(self, clip_actions=False)
        n_vars = observation_space["gamevars"].shape[0]
        self.cnn, self.trunk, h = _make_backbone(n_vars, hidden)
        self.head = nn.Linear(h, 1)

    def compute(self, inputs, role):
        obs = inputs.get("observations", inputs.get("states"))
        d = unflatten_tensorized_space(self.observation_space, obs)
        screen = d["screen"].float() / 255.0
        gv = d["gamevars"].float()
        feat = self.cnn(screen)
        h = self.trunk(torch.cat([feat, gv], dim=-1))
        return self.head(h), {}


@dataclass
class SkrlPPOConfig:
    total_steps: int = 500_000
    rollout: int = 256
    learning_epochs: int = 4
    mini_batches: int = 8
    lr: float = 2.5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.02
    vf_coef: float = 0.5
    log_interval: int = 50
    ckpt_interval: int = 25_000
    log_dir: str = "runs/skrl"
    num_envs: int = 4
    resume_from: str | None = None
    mixed_precision: bool = False


def train(cfg: SkrlPPOConfig, env_kwargs: dict | None = None, seed: int = 0) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    env_kwargs = env_kwargs or {}
    if cfg.num_envs > 1:
        raw_env = VecVizDoomMultiAgentEnv(
            num_envs=cfg.num_envs, num_players=2, seed=seed, **env_kwargs
        )
    else:
        raw_env = VizDoomMultiAgentEnv(num_players=2, seed=seed, **env_kwargs)
    env = wrap_env(raw_env, wrapper="pettingzoo")

    memories: dict[str, RandomMemory] = {}
    models: dict[str, dict[str, Model]] = {}
    for agent_name in env.possible_agents:
        memories[agent_name] = RandomMemory(
            memory_size=cfg.rollout, num_envs=env.num_envs, device=device,
        )
        obs_space = env.observation_spaces[agent_name]
        act_space = env.action_spaces[agent_name]
        models[agent_name] = {
            "policy": DoomActor(obs_space, act_space, device).to(device),
            "value": DoomCritic(obs_space, act_space, device).to(device),
        }

    agent_ids = list(env.possible_agents)
    ippo_cfg = IPPO_CFG(
        experiment=ExperimentCfg(
            directory=cfg.log_dir,
            experiment_name="ippo",
            write_interval=cfg.log_interval,
            checkpoint_interval=cfg.ckpt_interval,
        ),
        rollouts=cfg.rollout,
        learning_epochs=cfg.learning_epochs,
        mini_batches=cfg.mini_batches,
        discount_factor=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        learning_rate=cfg.lr,
        learning_rate_scheduler_kwargs={uid: {} for uid in agent_ids},
        observation_preprocessor_kwargs={uid: {} for uid in agent_ids},
        state_preprocessor_kwargs={uid: {} for uid in agent_ids},
        value_preprocessor_kwargs={uid: {} for uid in agent_ids},
        grad_norm_clip=0.5,
        ratio_clip=cfg.clip_eps,
        value_clip=0.2,
        entropy_loss_scale=cfg.ent_coef,
        value_loss_scale=cfg.vf_coef,
        mixed_precision=cfg.mixed_precision,
    )

    agent = IPPO(
        possible_agents=env.possible_agents,
        models=models,
        memories=memories,
        cfg=ippo_cfg,
        observation_spaces=env.observation_spaces,
        state_spaces=env.state_spaces,
        action_spaces=env.action_spaces,
        device=device,
    )

    if cfg.resume_from:
        agent.load(cfg.resume_from)
        # agent.load() calls .eval() on loaded modules; put them back to train
        for agent_name in env.possible_agents:
            for m in models[agent_name].values():
                m.train()
        print(f"[resume] loaded checkpoint from {cfg.resume_from}")

    trainer_cfg = {"timesteps": cfg.total_steps, "headless": True}
    trainer = SequentialTrainer(cfg=trainer_cfg, env=env, agents=agent)
    trainer.train()

    env.close()
