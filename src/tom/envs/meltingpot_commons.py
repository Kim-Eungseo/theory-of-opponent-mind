"""Self-contained re-implementation of MeltingPot's ``commons_harvest`` substrate.

DeepMind's real MeltingPot rides on ``dmlab2d``, which only ships binary wheels
for cpython 3.10/3.11 and otherwise needs a heavy Bazel/DMLab2D source build —
neither is available on this project's Python 3.13 venv. So, in the spirit of
the other lightweight env wrappers in this package (``overcooked_multi`` wraps
overcooked-ai, ``hanabi_multi`` wraps HLE), this module re-implements the
*mechanics* of the canonical ``commons_harvest__open`` substrate in pure NumPy.

What it preserves (the bits that matter for opponent-/partner-modeling research):

* **Spatially embodied, partially observed.** Each of N agents lives on a 2-D
  grid and sees only an egocentric ``(2R+1)×(2R+1)`` window, rotated so its own
  facing points "up" (exactly MeltingPot's egocentric convention).
* **The social dilemma.** Apples regrow stochastically at a rate that *grows
  with the local density of un-eaten apples* (Perolat et al. 2017 schedule). A
  patch harvested down to zero never recovers — so greedy over-harvesting is
  individually tempting but collectively ruinous (the tragedy of the commons).
* **An exclusion mechanism.** Agents can fire a short "zap" beam in their facing
  direction; a hit co-player is frozen and removed for a spell, then respawns.
  This is reward-neutral but lets an agent defend a patch (territoriality).

Two metrics are reported per episode, matching the MeltingPot literature:
collective return (sum of apples eaten by all players) and equality
(``1 - Gini`` of per-player returns).

API mirrors the other envs here: :class:`MeltingPotCommonsEnv` exposes a
PettingZoo-parallel-style dict API (``player_0 .. player_{N-1}``), and
:class:`VecMeltingPotCommonsEnv` runs ``num_envs`` games sequentially with
auto-reset, returning batched arrays with an explicit agent axis for the
shared-parameter IPPO trainer.
"""
from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

# ---- action set (move/turn/fire are separate, MeltingPot-style) --------
NOOP = 0
FORWARD = 1
BACKWARD = 2
STEP_LEFT = 3
STEP_RIGHT = 4
TURN_LEFT = 5
TURN_RIGHT = 6
FIRE_ZAP = 7
N_ACTIONS = 8
MOVE_ACTIONS = (FORWARD, BACKWARD, STEP_LEFT, STEP_RIGHT)

# facing: 0=N 1=E 2=S 3=W ; (drow, dcol)
_DIR = np.array([(-1, 0), (0, 1), (1, 0), (0, -1)], dtype=np.int64)

# observation channels of the egocentric window
CH_WALL, CH_APPLE, CH_OTHER, CH_BEAM = 0, 1, 2, 3
N_CHANNELS = 4

# Canonical open orchard: 4 diamond patches of apples on an open field.
DEFAULT_MAP = """
WWWWWWWWWWWWWWWW
W              W
W   AAA   AAA  W
W  AAAAA AAAAA W
W   AAA   AAA  W
W              W
W   AAA   AAA  W
W  AAAAA AAAAA W
W   AAA   AAA  W
W              W
W              W
WWWWWWWWWWWWWWWW
"""

MAPS = {"default": DEFAULT_MAP}


def _parse_map(ascii_map: str) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]]]:
    """Return (wall[H,W] bool, apple_cells, floor_cells)."""
    rows = [r for r in ascii_map.strip("\n").split("\n")]
    width = max(len(r) for r in rows)
    rows = [r.ljust(width, "W") for r in rows]  # pad ragged lines as wall
    H, W = len(rows), width
    wall = np.zeros((H, W), dtype=bool)
    apple_cells: list[tuple[int, int]] = []
    floor_cells: list[tuple[int, int]] = []
    for r in range(H):
        for c in range(W):
            ch = rows[r][c]
            if ch == "W":
                wall[r, c] = True
            elif ch == "A":
                apple_cells.append((r, c))
            else:
                floor_cells.append((r, c))
    return wall, apple_cells, floor_cells


def _gini(x: np.ndarray) -> float:
    """Gini coefficient of a non-negative vector; 0 when all equal/zero."""
    x = np.asarray(x, dtype=np.float64)
    if x.sum() <= 0:
        return 0.0
    x = np.sort(x)
    n = x.size
    idx = np.arange(1, n + 1)
    return float((np.sum((2 * idx - n - 1) * x)) / (n * x.sum()))


class MeltingPotCommonsEnv:
    """One Commons-Harvest game with an egocentric dict API."""

    metadata = {"name": "meltingpot_commons_v0"}

    def __init__(
        self,
        num_players: int = 5,
        map_name: str = "default",
        horizon: int = 1000,
        view_radius: int = 5,
        regrow_radius: int = 2,
        beam_length: int = 3,
        freeze_steps: int = 25,
        seed: int | None = None,
    ):
        self.num_players = int(num_players)
        self.horizon = int(horizon)
        self.R = int(view_radius)
        self.regrow_radius = int(regrow_radius)
        self.beam_length = int(beam_length)
        self.freeze_steps = int(freeze_steps)
        self._rng = np.random.default_rng(seed)

        self.wall, self.apple_cells, self.floor_cells = _parse_map(MAPS[map_name])
        self.H, self.W = self.wall.shape
        self.num_apples = len(self.apple_cells)
        self._apple_index = {cell: i for i, cell in enumerate(self.apple_cells)}
        self._apple_rc = np.array(self.apple_cells, dtype=np.int64)  # (A, 2)
        self._floor_rc = np.array(self.floor_cells, dtype=np.int64)

        # precompute, for each apple cell, the indices of other apple cells
        # within the Chebyshev regrow radius (drives density-dependent regrowth)
        self._neighbors: list[np.ndarray] = []
        for r, c in self.apple_cells:
            d = np.maximum(
                np.abs(self._apple_rc[:, 0] - r), np.abs(self._apple_rc[:, 1] - c)
            )
            nb = np.where((d > 0) & (d <= self.regrow_radius))[0]
            self._neighbors.append(nb)

        self.n_actions = N_ACTIONS
        win = 2 * self.R + 1
        self.obs_shape = (N_CHANNELS, win, win)
        self.obs_dim = int(np.prod(self.obs_shape))

        self.possible_agents = [f"player_{i}" for i in range(self.num_players)]
        self.agents = list(self.possible_agents)
        self.observation_spaces = {
            a: gym.spaces.Box(0.0, 1.0, self.obs_shape, np.float32)
            for a in self.possible_agents
        }
        self.action_spaces = {
            a: gym.spaces.Discrete(self.n_actions) for a in self.possible_agents
        }

        # state (filled by reset)
        self.apple_present = np.zeros(self.num_apples, dtype=bool)
        self.agent_pos = np.zeros((self.num_players, 2), dtype=np.int64)
        self.agent_facing = np.zeros(self.num_players, dtype=np.int64)
        self.agent_alive = np.ones(self.num_players, dtype=bool)
        self.freeze_timer = np.zeros(self.num_players, dtype=np.int64)
        self._beam = np.zeros((self.H, self.W), dtype=bool)
        self._ep_return = np.zeros(self.num_players, dtype=np.float64)
        self.t = 0

    # ---- internal array API (used by the vec wrapper) ----
    def _reset_arr(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.apple_present[:] = True
        self.agent_alive[:] = True
        self.freeze_timer[:] = 0
        self.agent_facing[:] = self._rng.integers(0, 4, size=self.num_players)
        spawn = self._rng.choice(len(self.floor_cells), size=self.num_players, replace=False)
        self.agent_pos[:] = self._floor_rc[spawn]
        self._beam[:] = False
        self._ep_return[:] = 0.0
        self.t = 0
        self.agents = list(self.possible_agents)
        return self._all_obs()

    def _random_spawn(self) -> tuple[int, int]:
        occ = {tuple(self.agent_pos[i]) for i in range(self.num_players) if self.agent_alive[i]}
        order = self._rng.permutation(len(self.floor_cells))
        for j in order:
            cell = tuple(self._floor_rc[j])
            if cell not in occ:
                return cell
        return tuple(self._floor_rc[order[0]])  # degenerate fallback

    def _step_arr(self, actions: np.ndarray):
        actions = np.asarray(actions, dtype=np.int64).reshape(-1)
        rewards = np.zeros(self.num_players, dtype=np.float32)
        self._beam[:] = False

        # 1. respawn agents whose freeze elapsed
        for i in range(self.num_players):
            if not self.agent_alive[i]:
                self.freeze_timer[i] -= 1
                if self.freeze_timer[i] <= 0:
                    self.agent_pos[i] = self._random_spawn()
                    self.agent_alive[i] = True

        # 2. turns
        for i in range(self.num_players):
            if not self.agent_alive[i]:
                continue
            if actions[i] == TURN_LEFT:
                self.agent_facing[i] = (self.agent_facing[i] - 1) % 4
            elif actions[i] == TURN_RIGHT:
                self.agent_facing[i] = (self.agent_facing[i] + 1) % 4

        # 3. moves with claim-based conflict resolution (no swaps / no stacking)
        occupied = {
            tuple(self.agent_pos[i]): i
            for i in range(self.num_players)
            if self.agent_alive[i]
        }
        movers = [
            i for i in range(self.num_players)
            if self.agent_alive[i] and actions[i] in MOVE_ACTIONS
        ]
        for i in self._rng.permutation(movers) if movers else []:
            delta = self._move_delta(int(actions[i]), int(self.agent_facing[i]))
            r0, c0 = int(self.agent_pos[i, 0]), int(self.agent_pos[i, 1])
            tr, tc = r0 + delta[0], c0 + delta[1]
            if 0 <= tr < self.H and 0 <= tc < self.W and not self.wall[tr, tc] and (tr, tc) not in occupied:
                del occupied[(r0, c0)]
                occupied[(tr, tc)] = i
                self.agent_pos[i] = (tr, tc)

        # 4. collection (resolved positions; one agent per cell ⇒ no double pick)
        for i in range(self.num_players):
            if not self.agent_alive[i]:
                continue
            idx = self._apple_index.get(tuple(self.agent_pos[i]))
            if idx is not None and self.apple_present[idx]:
                self.apple_present[idx] = False
                rewards[i] += 1.0

        # 5. zaps (use post-move positions/facings)
        zapped: set[int] = set()
        for i in range(self.num_players):
            if not self.agent_alive[i] or actions[i] != FIRE_ZAP:
                continue
            dr, dc = _DIR[self.agent_facing[i]]
            r, c = int(self.agent_pos[i, 0]), int(self.agent_pos[i, 1])
            for _ in range(self.beam_length):
                r += int(dr); c += int(dc)
                if not (0 <= r < self.H and 0 <= c < self.W) or self.wall[r, c]:
                    break
                self._beam[r, c] = True
                tgt = occupied.get((r, c))
                if tgt is not None and tgt != i and self.agent_alive[tgt]:
                    zapped.add(tgt)
        for t in zapped:
            self.agent_alive[t] = False
            self.freeze_timer[t] = self.freeze_steps

        # 6. density-dependent apple regrowth
        self._regrow()

        # 7. bookkeeping
        self._ep_return += rewards
        self.t += 1
        truncated = self.t >= self.horizon
        obs = self._all_obs()
        return obs, rewards, truncated

    def _move_delta(self, action: int, facing: int) -> np.ndarray:
        if action == FORWARD:
            return _DIR[facing]
        if action == BACKWARD:
            return _DIR[(facing + 2) % 4]
        if action == STEP_LEFT:
            return _DIR[(facing - 1) % 4]
        if action == STEP_RIGHT:
            return _DIR[(facing + 1) % 4]
        return np.array([0, 0], dtype=np.int64)

    def _regrow(self) -> None:
        empty = np.where(~self.apple_present)[0]
        if empty.size == 0:
            return
        probs = np.empty(empty.size, dtype=np.float64)
        for k, i in enumerate(empty):
            n = int(self.apple_present[self._neighbors[i]].sum())
            probs[k] = _regrow_prob(n)
        draws = self._rng.random(empty.size)
        self.apple_present[empty[draws < probs]] = True

    # ---- observation ----
    def _layers(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        apple = np.zeros((self.H, self.W), dtype=np.float32)
        present = self._apple_rc[self.apple_present]
        if present.size:
            apple[present[:, 0], present[:, 1]] = 1.0
        agents = np.zeros((self.H, self.W), dtype=np.float32)
        for i in range(self.num_players):
            if self.agent_alive[i]:
                agents[self.agent_pos[i, 0], self.agent_pos[i, 1]] = 1.0
        return apple, agents, self._beam.astype(np.float32)

    def _all_obs(self) -> np.ndarray:
        """(num_players, C, H_win, W_win) egocentric, rotated to each facing."""
        R = self.R
        apple, agents, beam = self._layers()
        # pad: walls outside the map, zeros elsewhere
        wall_p = np.pad(self.wall.astype(np.float32), R, constant_values=1.0)
        apple_p = np.pad(apple, R, constant_values=0.0)
        agents_p = np.pad(agents, R, constant_values=0.0)
        beam_p = np.pad(beam, R, constant_values=0.0)
        win = 2 * R + 1
        out = np.zeros((self.num_players, N_CHANNELS, win, win), dtype=np.float32)
        for i in range(self.num_players):
            if not self.agent_alive[i]:
                out[i, CH_WALL] = 1.0  # frozen/off-board: see only walls
                continue
            r, c = int(self.agent_pos[i, 0]), int(self.agent_pos[i, 1])
            sl = (slice(r, r + win), slice(c, c + win))
            crop = np.stack([wall_p[sl], apple_p[sl], agents_p[sl], beam_p[sl]])
            crop[CH_OTHER, R, R] = 0.0  # center is self, not "other"
            # rotate so this agent's facing points up (k == facing index)
            out[i] = np.rot90(crop, k=int(self.agent_facing[i]), axes=(-2, -1))
        return out

    def _episode_stats(self) -> dict[str, float]:
        coll = float(self._ep_return.sum())
        return {
            "collective_return": coll,
            "per_capita_return": coll / self.num_players,
            "equality": 1.0 - _gini(self._ep_return),
            "apples_remaining": float(self.apple_present.sum()),
            "length": int(self.t),
        }

    # ---- public dict API (parity with the other envs) ----
    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
        obs = self._reset_arr(seed=seed)
        return (
            {a: obs[i] for i, a in enumerate(self.possible_agents)},
            {a: {} for a in self.possible_agents},
        )

    def step(self, actions: dict[str, int]):
        a_arr = np.array([actions[a] for a in self.possible_agents], dtype=np.int64)
        obs, rew, truncated = self._step_arr(a_arr)
        obs_d = {a: obs[i] for i, a in enumerate(self.possible_agents)}
        rew_d = {a: float(rew[i]) for i, a in enumerate(self.possible_agents)}
        term_d = {a: False for a in self.possible_agents}
        trunc_d = {a: bool(truncated) for a in self.possible_agents}
        stats = self._episode_stats() if truncated else {}
        info_d = {a: dict(stats) for a in self.possible_agents}
        return obs_d, rew_d, term_d, trunc_d, info_d

    def state(self) -> np.ndarray:
        """Global allocentric state: stacked wall/apple/agent layers (C,H,W)."""
        apple, agents, _ = self._layers()
        return np.stack([self.wall.astype(np.float32), apple, agents])

    def close(self) -> None:
        pass


def _regrow_prob(n: int) -> float:
    """Apple regrowth probability given ``n`` apples within the regrow radius.

    Perolat et al. (2017) common-pool schedule: a locally depleted patch
    (n == 0) can never recover, which is what makes over-harvesting a trap.
    """
    if n <= 0:
        return 0.0
    if n <= 2:
        return 0.01
    if n <= 4:
        return 0.05
    return 0.1


class VecMeltingPotCommonsEnv:
    """``num_envs`` Commons-Harvest games stepped sequentially with auto-reset.

    Pure-Python grid logic is cheap enough that a for-loop beats IPC overhead
    (same rationale as :class:`VecOvercookedEnv`). Observations/rewards carry an
    explicit agent axis — ``(num_envs, num_players, ...)`` — because the trainer
    shares one set of policy parameters across all players.
    """

    def __init__(
        self,
        num_envs: int = 16,
        num_players: int = 5,
        map_name: str = "default",
        horizon: int = 1000,
        view_radius: int = 5,
        regrow_radius: int = 2,
        beam_length: int = 3,
        freeze_steps: int = 25,
        seed: int = 0,
    ):
        self._num_envs = int(num_envs)
        self.envs = [
            MeltingPotCommonsEnv(
                num_players=num_players,
                map_name=map_name,
                horizon=horizon,
                view_radius=view_radius,
                regrow_radius=regrow_radius,
                beam_length=beam_length,
                freeze_steps=freeze_steps,
                seed=seed + i,
            )
            for i in range(num_envs)
        ]
        tpl = self.envs[0]
        self.num_players = tpl.num_players
        self.possible_agents = tpl.possible_agents
        self.observation_spaces = tpl.observation_spaces
        self.action_spaces = tpl.action_spaces
        self.obs_shape = tpl.obs_shape
        self.obs_dim = tpl.obs_dim
        self.n_actions = tpl.n_actions
        self._last_obs = np.zeros((num_envs, tpl.num_players, *tpl.obs_shape), dtype=np.float32)

    @property
    def num_envs(self) -> int:
        return self._num_envs

    def reset(self, seed: int | None = None) -> np.ndarray:
        for i, e in enumerate(self.envs):
            self._last_obs[i] = e._reset_arr(seed=None if seed is None else seed + i)
        return self._last_obs.copy()

    def step(self, actions: np.ndarray):
        """actions: (num_envs, num_players) int array."""
        actions = np.asarray(actions, dtype=np.int64).reshape(self._num_envs, self.num_players)
        n, P = self._num_envs, self.num_players
        obs = np.zeros((n, P, *self.obs_shape), dtype=np.float32)
        rew = np.zeros((n, P), dtype=np.float32)
        term = np.zeros((n, P), dtype=bool)
        trunc = np.zeros((n, P), dtype=bool)
        completed: list[dict | None] = [None] * n
        for i in range(n):
            o_i, r_i, truncated = self.envs[i]._step_arr(actions[i])
            obs[i] = o_i
            rew[i] = r_i
            trunc[i] = truncated
            if truncated:
                completed[i] = self.envs[i]._episode_stats()
                obs[i] = self.envs[i]._reset_arr()  # auto-reset
        self._last_obs = obs.copy()
        return obs, rew, term, trunc, {"completed": completed}

    def close(self) -> None:
        for e in self.envs:
            e.close()
