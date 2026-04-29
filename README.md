# Theory of Opponent Mind

Multi-environment cooperative-MARL sandbox for **opponent / partner modeling**
research, with self-play baselines and OM-aux variants implemented across
three environments (Overcooked, Hanabi, ViZDoom).

The work is organised into three conda environments because their RL
frameworks pull in mutually-incompatible dependencies (gym/gymnasium, ray,
TF version, PyTorch CUDA build). Pick the env that matches what you want
to run.

| Env | Domain | Framework | Status |
|-----|--------|-----------|--------|
| `tom-coop` | Overcooked + Hanabi (cooperative) | PyTorch + own IPPO | **active** |
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
│   └── hanabi_multi.py         # turn-based 2-player Hanabi (HLE)
└── training/
    ├── ippo_overcooked.py      # IPPO + OM/SOM/TOM aux + BAD routing
    ├── ippo_hanabi.py          # shared-policy PPO + belief aux + BAD routing
    ├── ippo_hanabi_lstm.py     # recurrent variant (true BPTT)
    └── skrl_ppo.py             # legacy ViZDoom IPPO via skrl

scripts/
├── train_overcooked.py         # main Overcooked CLI
├── train_hanabi.py             # main Hanabi CLI
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
