# Theory of Opponent Mind

Multi-agent ViZDoom RL sandbox for opponent modeling research.

- PettingZoo `ParallelEnv` wrapping a ViZDoom multiplayer deathmatch (one OS process per player).
- Self-play PPO baseline (single-file, PyTorch).
- Match recording to mp4 (side-by-side per-player view).

## Requirements

- Linux (tested on Ubuntu 24.04).
- `conda` / `miniconda`.
- NVIDIA GPU recommended (Blackwell = PyTorch nightly required; see below).

## Install

```bash
# 1. create env
conda create -n tom python=3.11 -y
conda activate tom

# 2. core deps
pip install vizdoom gymnasium pettingzoo numpy opencv-python

# 3. pytorch
#  - Ampere / Ada Lovelace GPUs (sm_80 / sm_89):
pip install torch --index-url https://download.pytorch.org/whl/cu124
#  - Blackwell GPUs (sm_120, e.g. RTX PRO 4500):
pip install --pre --upgrade torch --index-url https://download.pytorch.org/whl/nightly/cu128

# 4. (optional) dev tools
pip install pytest pytest-timeout
```

## Quick sanity check

```bash
# single-player ViZDoom works
python scripts/smoke_test.py

# two agents, random policies, print match stats
PYTHONPATH=src python scripts/random_match.py --players 2 --seconds 30

# pytest
PYTHONPATH=src pytest tests/
```

## Scripts

All scripts expect `PYTHONPATH=src`.

| script | what it does |
|---|---|
| `scripts/smoke_test.py` | Single-player ViZDoom `basic.cfg` episode — validates install |
| `scripts/random_match.py` | 2-agent random match, prints frags/returns |
| `scripts/record_match.py` | Record match to mp4 (random or from `--ckpt`) |
| `scripts/train.py` | Self-play PPO trainer with optional checkpoint saving |
| `scripts/minimal_mp.py` | Bare-minimum multiprocessing multiplayer reference |

### Examples

```bash
# record a 30s random CIG match at real game speed
PYTHONPATH=src python scripts/record_match.py \
    --scenario cig --out videos/cig_random.mp4 \
    --seconds 30 --frame-skip 1 --fps 35

# train 100k env-steps and save checkpoint
PYTHONPATH=src python scripts/train.py \
    --total-steps 100000 --save checkpoints/ppo.pt

# record with trained policy
PYTHONPATH=src python scripts/record_match.py \
    --ckpt checkpoints/ppo.pt --out videos/trained.mp4 --seconds 30
```

## Project layout

```
src/tom/
├── envs/vizdoom_multi.py      # PettingZoo ParallelEnv, ASYNC multi-process
├── agents/random_agent.py
└── training/ppo.py            # single-file self-play PPO

scripts/                       # entry points (see table above)
tests/test_env_smoke.py        # pytest smoke test
```

## Notes & gotchas

- **Default scenario is `multi_duel.cfg`** (1v1). For richer maps try `--scenario cig` or `--scenario multi`. `deathmatch.cfg` is single-player only and will crash on host/join.
- **One OS process per player.** ViZDoom's networked runtime isn't thread-safe in-process.
- **`ASYNC_PLAYER` mode is used** — strict `PLAYER` (tick-sync) deadlocks under pipe-driven stepping.
- **`ticrate=1000`** by default so env runs ~28× wall-clock. Use `--ticrate 35` for real-time playback-matched runs.
- **`frame_skip=4`** for training (RL convention), **`frame_skip=1`** for smooth real-time recording.
- Binary action space (7 buttons) — no delta turning / mouselook yet. Adding `MultiDiscrete` with delta buttons is a follow-up for finer aim.
