# Theory of Opponent Mind

Multi-environment cooperative-MARL sandbox for **opponent / partner modeling**
research, with self-play baselines and OM-aux variants implemented across
four environments (Overcooked, Hanabi, MeltingPot, ViZDoom).

The work is organised into three conda environments because their RL
frameworks pull in mutually-incompatible dependencies (gym/gymnasium, ray,
TF version, PyTorch CUDA build). Pick the env that matches what you want
to run.

| Env | Domain | Framework | Status |
|-----|--------|-----------|--------|
| `tom-coop` | Overcooked + Hanabi + MeltingPot (NumPy reimpl) | PyTorch + own IPPO | **active** |
| `tom-meltingpot` | MeltingPot (real, dmlab2d substrates) | PyTorch + own IPPO | **active** |
| `tom-sf`   | ViZDoom 1v1 (adversarial) | Sample-Factory | dormant |
| `tom-carroll` | Overcooked Carroll-2019 reference | RLLib + TF | reproduction-only |

---

## Quick install (active env)

If you only want to reproduce the cooperative OM experiments (Overcooked +
Hanabi), this is the only env you need.

```bash
conda create -n tom-coop python=3.10 -y
conda activate tom-coop

pip install --upgrade pip
pip install overcooked-ai==1.1.0 hanabi-learning-environment
pip install torch torchvision tensorboard tqdm
pip install scipy gymnasium "numpy<2"
```

`numpy<2` is required because `overcooked-ai 1.1.0` calls the removed
`np.Inf`. CPU-only PyTorch is fine — the policies are tiny MLPs.

---

## Per-environment install

### `tom-coop` — Overcooked + Hanabi (PyTorch)

```bash
conda create -n tom-coop python=3.10 -y
conda activate tom-coop

pip install --upgrade pip
pip install overcooked-ai==1.1.0
pip install hanabi-learning-environment           # 2-player Hanabi (HLE)
pip install torch torchvision tensorboard tqdm
pip install scipy gymnasium "numpy<2"
```

Smoke check:

```bash
PYTHONPATH=src python scripts/train_overcooked.py \
    --layout asymmetric_advantages --total-steps 50000 \
    --num-envs 8 --rollout 200 --log-dir runs_overcooked/smoke
PYTHONPATH=src python scripts/train_hanabi.py \
    --total-steps 50000 --num-envs 8 --rollout-steps 64 --log-dir runs_hanabi/smoke
```

### MeltingPot Commons-Harvest — backend A: pure-NumPy reimplementation (no extra deps)

A self-contained re-implementation of MeltingPot's canonical
`commons_harvest__open` substrate — a *tragedy-of-the-commons* social dilemma.
DeepMind's `dm-meltingpot` rides on `dmlab2d`, which ships no wheels for
Python ≥ 3.12 and otherwise needs a heavy Bazel/DMLab2D source build, so this
substrate is written in pure NumPy and needs **only `numpy` + `torch` +
`tensorboard`** — it runs in the `tom-coop` env as-is, no install step.

N agents harvest apples on a grid under egocentric partial observation; apples
regrow at a rate that grows with local apple density (Perolat et al. 2017), so
a patch harvested to zero never recovers. Baseline = shared-parameter IPPO with
a small conv actor-critic. Reported metrics: collective return (Σ apples) and
equality (`1 - Gini`).

Smoke check (~1 min on GPU):

```bash
PYTHONPATH=src python scripts/train_meltingpot.py \
    --num-players 5 --num-envs 8 --horizon 200 \
    --total-steps 80000 --rollout 100 --log-dir runs_meltingpot/smoke
```

Full example run (1M env-steps, faithful horizon=1000). Collective return rises
to a peak, then the commons collapses as the selfish shared policy
over-harvests — the dilemma the substrate is built to expose:

```bash
PYTHONPATH=src python scripts/train_meltingpot.py \
    --num-players 5 --num-envs 16 --horizon 1000 \
    --total-steps 1000000 --seed 0 \
    --log-dir runs_meltingpot/ippo_commons_default_p5
```

Observed curve (seed 0, ~37 s on a single GPU @ ~27k steps/s):

| env-steps | collective return | equality | apples left | phase |
|---:|---:|---:|---:|---|
| (random) | ~119 | 0.74 | ~40 | untrained reference |
| 152k | **918** | 0.93 | 22 | peak — efficient harvesting learned |
| 352k | 656 | 0.92 | 5 | over-harvesting begins |
| 512k | 281 | 0.88 | 0 | **commons collapsed** (local `n=0` ⇒ no regrowth) |
| 1.0M | 202 | 0.87 | 0 | converged to the depleted equilibrium |

The selfish shared policy learns to harvest (return ≫ random), then drives the
commons to collapse — exactly the dilemma `commons_harvest` is built to expose.
The mid-training checkpoint `ckpt_000504000.pt` holds the *high-welfare,
pre-collapse* policy; `ckpt_final.pt` holds the depleted-equilibrium one. This
is the motivating baseline for partner-aware variants (add the OM/SOM/TOM aux
heads from `ippo_overcooked` to the conv trunk to study whether modeling
co-players' harvest intent can sustain the commons).

### MeltingPot Commons-Harvest — backend B: the real `dm-meltingpot`

The genuine DeepMind substrate (dmlab2d-backed). Because `dmlab2d` only ships
wheels for cpython 3.10/3.11, it lives in its own conda env:

```bash
conda create -n tom-meltingpot python=3.11 -y
conda activate tom-meltingpot
pip install --only-binary=dmlab2d dm-meltingpot      # pulls dmlab2d + bundled assets (~0.5 GB wheel)
pip install torch --index-url https://download.pytorch.org/whl/cu128   # Blackwell sm_120; pick your CUDA
pip install tensorboard gymnasium
```

`src/tom/envs/meltingpot_substrate.py` wraps any of the 49 substrates behind the
*same* array API as the NumPy version, so the IPPO trainer
(`tom.training.ippo_meltingpot.train`) runs unchanged — only the observation
(real egocentric **88×88×3 RGB** sprites) and the conv backbone
(`NatureActorCritic`, a Nature-CNN) differ. `commons_harvest__open` runs
**7 players**, 8 actions, ~7.7k agent-steps/s on a single GPU.

```bash
conda run -n tom-meltingpot env PYTHONPATH=src TF_CPP_MIN_LOG_LEVEL=3 \
    python scripts/train_meltingpot_real.py \
    --substrate commons_harvest__open --num-envs 6 \
    --total-steps 1000000 --rollout 100 --seed 0
```

Observed (seed 0, 6 envs, ~6.5k agent-steps/s, **~3 min for 1M steps** on one GPU):

| env-steps | collective | per-capita | equality |
|---:|---:|---:|---:|
| 113k | 206 (peak) | 29.5 | 0.59 |
| 504k | 130 | 18.6 | 0.71 |
| 1.0M | 86 | 12.3 | 0.75 |

At 1M steps the policy is still near-random (`loss/ent` ≈ −2.0 vs the ln 8 = 2.08
maximum; `kl` ≈ 0.001), so vanilla IPPO underfits and collective return drifts
*down* as the commons depletes. That is the expected scale story — **1M
agent-steps ≈ only ~140 episodes** of 88×88 RGB across 7 players. Treat this as a
correctly-wired, runnable baseline, not a tuned result.

> Real MeltingPot is far harder than the NumPy toy (RGB perception, 7 players,
> 1000-step episodes); 1M steps is small by MeltingPot standards (published
> baselines use 100M–1B steps + population-based training), so expect a
> clear-but-modest upward trend, not state-of-the-art numbers.

### `tom-sf` — ViZDoom 1v1 (Sample-Factory)

Sample-Factory needs a recent torch build. For Blackwell GPUs (sm_120) the
nightly cu128 wheel is required.

```bash
conda create -n tom-sf python=3.10 -y
conda activate tom-sf

pip install --upgrade pip
# Ampere / Ada Lovelace (sm_80 / sm_89):
pip install torch --index-url https://download.pytorch.org/whl/cu124
# Blackwell (sm_120, e.g. RTX PRO 4500):
pip install --pre --upgrade torch --index-url https://download.pytorch.org/whl/nightly/cu128

pip install vizdoom gymnasium pettingzoo opencv-python
pip install -e external/sample-factory
```

Smoke check:

```bash
PYTHONPATH=src python scripts/smoke_test.py
PYTHONPATH=src python scripts/random_match.py --players 2 --seconds 30
```

### `tom-carroll` — Carroll 2019 RLLib reference (TF + ray)

Heavyweight, only needed if you want to reproduce the Carroll-style PPO+BC
baseline on Overcooked. The ray 2.2 + TF 2.10 stack does not run on
Blackwell GPUs (no Blackwell-compatible CUDA libs at this TF version), so
this env runs CPU-only.

```bash
conda create -n tom-carroll python=3.10 -y
conda activate tom-carroll

pip install --upgrade pip
pip install -e external/overcooked_ai[harl]            # ray 2.2 + TF 2.19 + sacred etc.
pip install sacred pymongo
pip install "setuptools<81"                            # sacred needs pkg_resources
pip install "pydantic<2"                               # ray 2.2 incompatible with pydantic 2
pip install "tensorflow==2.10" "tensorflow-probability<0.18"
pip install "numpy==1.23.5"                            # ray pickled RNG breaks on numpy>=1.24
```

Two source patches are required (these are stored applied in `external/`,
but if you reinstall they need to be re-applied):

1. `ray/rllib/utils/pre_checks/env.py`: replace `(bool, np.bool, np.bool_)`
   with `(bool, np.bool_)` (np.bool was removed in newer numpy).
2. `external/overcooked_ai/src/human_aware_rl/rllib/rllib.py` — `on_train_result`
   accepts both `trainer=` (old ray) and `algorithm=` (ray ≥ 2.0):

   ```python
   def on_train_result(self, *, algorithm=None, result=None, trainer=None, **kwargs):
       if trainer is None:
           trainer = algorithm
       ...  # rest of method unchanged
   ```

Smoke check (3 iters, ~25 s on CPU):

```bash
cd external/overcooked_ai/src/human_aware_rl/ppo
python ppo_rllib_client.py with \
    layout_name=asymmetric_advantages num_training_iters=3 \
    num_workers=4 num_gpus=0 verbose=False \
    results_dir=$PWD/../../../../../../runs_carroll \
    temp_dir=/tmp/ray_carroll_smoke
```

---

## Project layout

```
src/tom/
├── envs/
│   ├── vizdoom_multi.py        # PettingZoo ParallelEnv, multiprocess
│   ├── vec_vizdoom.py          # vectorised wrapper for ViZDoom
│   ├── overcooked_multi.py     # 2-agent Overcooked, dict API
│   ├── hanabi_multi.py         # turn-based 2-player Hanabi (HLE)
│   ├── meltingpot_commons.py   # N-agent Commons-Harvest (pure NumPy, no dmlab2d)
│   └── meltingpot_substrate.py # real dm-meltingpot substrates (dmlab2d, tom-meltingpot env)
└── training/
    ├── ippo_overcooked.py      # IPPO + OM/SOM/TOM aux + BAD routing
    ├── ippo_hanabi.py          # shared-policy PPO + belief aux + BAD routing
    ├── ippo_hanabi_lstm.py     # recurrent variant (true BPTT)
    ├── ippo_meltingpot.py      # shared-param IPPO; ConvActorCritic (NumPy) + NatureActorCritic (real)
    └── skrl_ppo.py             # legacy ViZDoom IPPO via skrl

scripts/
├── train_overcooked.py         # main Overcooked CLI
├── train_hanabi.py             # main Hanabi CLI
├── train_meltingpot.py         # MeltingPot Commons-Harvest CLI (NumPy backend)
├── train_meltingpot_real.py    # real dm-meltingpot CLI (tom-meltingpot env)
├── train_hanabi_lstm.py
├── probe_overcooked.py         # linear probing of OM info in encoder
├── train_skrl.py               # legacy ViZDoom training
├── record_match.py             # ViZDoom mp4 recorder
└── …

external/                       # vendored prior work (gitignored)
├── overcooked_ai/              # HumanCompatibleAI + human_aware_rl
├── HARL/, on-policy/, PantheonRL/
└── sample-factory/

runs_overcooked/, runs_hanabi/  # training output (gitignored)
runs_meltingpot/                # MeltingPot training output (gitignored)
runs_carroll/, runs_sf/, runs/
```

---

## Common training commands (`tom-coop`)

```bash
# vanilla self-play on asymmetric_advantages, 5M env-steps, seed 0
PYTHONPATH=src python scripts/train_overcooked.py \
    --layout asymmetric_advantages --total-steps 5000000 \
    --num-envs 32 --rollout 400 --seed 0 --log-dir runs_overcooked/aa_v_s0

# trajectory-OM with BAD-style routing (TOM+BAD)
PYTHONPATH=src python scripts/train_overcooked.py \
    --layout asymmetric_advantages --total-steps 5000000 \
    --num-envs 32 --rollout 400 --seed 0 \
    --tom-coef 0.3 --tom-in-policy --log-dir runs_overcooked/aa_tomb_s0

# capacity-matched wide-vanilla (h=400 ≈ 204K params)
PYTHONPATH=src python scripts/train_overcooked.py \
    --layout asymmetric_advantages --total-steps 5000000 \
    --num-envs 32 --rollout 400 --hidden 400 --seed 0 \
    --log-dir runs_overcooked/aa_wide_s0

# Hanabi shared-policy PPO with belief aux head
PYTHONPATH=src python scripts/train_hanabi.py \
    --total-steps 5000000 --num-envs 32 --rollout-steps 128 \
    --belief-coef 0.3 --seed 0 --log-dir runs_hanabi/abl_belief
```

TensorBoard: `tensorboard --logdir runs_overcooked` (or `runs_hanabi`).

---

## Method flags (Overcooked trainer)

| flag | what it adds |
|---|---|
| `--om-coef <c>` | partner-action prediction aux head (single-step, shared encoder) |
| `--om-in-policy` | concat OM softmax into policy/value head input (BAD-style) |
| `--som-coef <c>` | Self-Other Modeling — pass partner_obs through *my own* policy |
| `--tom-coef <c>` | trajectory OM head — separate LSTM over partner's last K obs |
| `--tom-history-len K` | history length for trajectory OM (default 8) |
| `--tom-in-policy` | concat trajectory-OM softmax into policy input (BAD-style) |
| `--hidden H` | encoder hidden width (raise to 400 for capacity-matched control) |
| `--ent-coef-end <e>` `--ent-coef-horizon <s>` | entropy coefficient anneal |

---

## Notes & gotchas

- **`numpy<2` is required for `overcooked-ai 1.1.0`** — it calls `np.Inf`.
- **One OS process per ViZDoom player** (`tom-sf`). The networked runtime is
  not thread-safe in-process.
- **`ASYNC_PLAYER` mode** is used in ViZDoom — strict `PLAYER` deadlocks under
  pipe-driven stepping.
- **`tom-carroll` is CPU-only on Blackwell**: TF 2.10 has no Blackwell CUDA
  build. Set `num_gpus=0` when launching the RLLib client.
- **Carroll patches**: if you re-install the env, re-apply the two source
  patches above before launching.
- The `external/` directory holds vendored prior work. It is gitignored —
  clone the upstream repos yourself if you want to update them.
