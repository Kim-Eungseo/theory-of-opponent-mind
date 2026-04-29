"""Multi-agent Hanabi wrapper around the DeepMind Hanabi-Learning-Environment.

Hanabi is **turn-based**: at any game step exactly one player acts. To plug
this into a shared-policy IPPO loop cleanly we expose a per-player MDP:

* ``step(action)`` advances the game one turn (the *current player's* turn).
* The next observation we return is from the *new* current player's
  perspective. The trainer treats each call as one transition for the player
  who just acted.

Reward attribution: Hanabi is fully cooperative, so the reward returned by
``env.step`` is added to *both* players' pending-transition accumulator. The
trainer reads ``info["reward_for"]`` to know which player just acted.

We expose a vectorised version (``VecHanabiEnv``) for FPS — Hanabi is pure
Python/C++ work that's cheap enough to run sequentially across N games.
"""
from __future__ import annotations

import numpy as np
from hanabi_learning_environment import rl_env

DEFAULT_ENV_NAME = "Hanabi-Full-CardKnowledge"

HAND_SIZE = 5
N_COLORS = 5
N_RANKS = 5
MISSING_CARD = -1  # sentinel: card slot is empty (deck depleted)


class HanabiMultiAgentEnv:
    """One Hanabi game, exposing a per-turn step API.

    ``reset()`` returns ``(obs_vec, legal_mask, current_player)``.
    ``step(action)`` returns the same triple plus reward & done.
    """

    metadata = {"name": "hanabi_multi_v0"}

    def __init__(
        self,
        env_name: str = DEFAULT_ENV_NAME,
        num_players: int = 2,
        seed: int | None = None,
    ):
        self.env_name = env_name
        self.num_players = num_players
        self._rng = np.random.default_rng(seed)
        self.env = rl_env.make(env_name, num_players=num_players)

        # introspect dims
        obs0 = self.env.reset()
        po0 = obs0["player_observations"][obs0["current_player"]]
        self.obs_dim = int(len(po0["vectorized"]))
        self.n_actions = int(self.env.num_moves())
        self.hand_size = HAND_SIZE
        self.n_colors = N_COLORS
        self.n_ranks = N_RANKS
        # restore — we don't want this to count as the "real" reset for the user
        self._cached_first_obs = obs0

    # ---- helpers ----
    def _extract(self, raw_obs) -> tuple[np.ndarray, np.ndarray, int]:
        cur = int(raw_obs["current_player"])
        po = raw_obs["player_observations"][cur]
        vec = np.asarray(po["vectorized"], dtype=np.float32)
        legal = np.zeros(self.n_actions, dtype=bool)
        for idx in po["legal_moves_as_int"]:
            legal[int(idx)] = True
        return vec, legal, cur

    def get_player_hands(self) -> np.ndarray:
        """Ground-truth hands for all players, shape (num_players, HAND_SIZE, 2).

        Returned as ``[..., 0] = color`` and ``[..., 1] = rank``. Empty slots
        (deck depleted near game end) are filled with ``MISSING_CARD``.
        """
        out = np.full((self.num_players, self.hand_size, 2), MISSING_CARD, dtype=np.int64)
        try:
            hands = self.env.state.player_hands()
        except Exception:
            return out
        for p, hand in enumerate(hands):
            for j, card in enumerate(hand[: self.hand_size]):
                out[p, j, 0] = int(card.color())
                out[p, j, 1] = int(card.rank())
        return out

    # ---- API ----
    def reset(self) -> tuple[np.ndarray, np.ndarray, int]:
        if self._cached_first_obs is not None:
            raw = self._cached_first_obs
            self._cached_first_obs = None
        else:
            raw = self.env.reset()
        return self._extract(raw)

    def step(self, action: int) -> tuple[np.ndarray, np.ndarray, int, float, bool, dict]:
        actor = None
        # who is acting NOW (before step)
        # (we know from the most recent obs but the env also tracks state internally)
        # We re-query via state to be safe. HLE doesn't expose current_player without
        # a fresh obs, so we rely on the caller to remember it. We return it via info.
        raw, reward, done, info = self.env.step(int(action))
        if not done:
            vec, legal, cur = self._extract(raw)
        else:
            # game over → no further obs/legal mask, but we return zeros
            vec = np.zeros(self.obs_dim, dtype=np.float32)
            legal = np.ones(self.n_actions, dtype=bool)
            cur = -1
        return vec, legal, cur, float(reward), bool(done), info or {}

    def close(self):
        pass


class VecHanabiEnv:
    """N independent Hanabi games stepped sequentially.

    Maintains per-env, per-player pending transitions and reward accumulators
    for the trainer. The trainer just supplies actions for whichever player
    is currently to act in each env, and gets back batched (obs, legal_mask,
    current_player) plus per-env step rewards/done flags.

    Auto-resets envs when an episode ends. Tracks per-episode score for
    logging.
    """

    def __init__(
        self,
        num_envs: int = 32,
        env_name: str = DEFAULT_ENV_NAME,
        num_players: int = 2,
        seed: int = 0,
    ):
        self._num_envs = int(num_envs)
        self.num_players = num_players
        self.envs = [
            HanabiMultiAgentEnv(env_name=env_name, num_players=num_players, seed=seed + i)
            for i in range(num_envs)
        ]
        tpl = self.envs[0]
        self.obs_dim = tpl.obs_dim
        self.n_actions = tpl.n_actions

        # per-env current state
        self._cur_obs = np.zeros((num_envs, tpl.obs_dim), dtype=np.float32)
        self._cur_legal = np.zeros((num_envs, tpl.n_actions), dtype=bool)
        self._cur_player = np.zeros(num_envs, dtype=np.int64)
        self._cur_hands = np.full((num_envs, num_players, tpl.hand_size, 2), MISSING_CARD, dtype=np.int64)
        self._ep_score = np.zeros(num_envs, dtype=np.float32)
        self._ep_len = np.zeros(num_envs, dtype=np.int32)
        self._completed: list[dict | None] = [None] * num_envs

    @property
    def hand_size(self) -> int:
        return self.envs[0].hand_size

    @property
    def n_colors(self) -> int:
        return self.envs[0].n_colors

    @property
    def n_ranks(self) -> int:
        return self.envs[0].n_ranks

    @property
    def num_envs(self) -> int:
        return self._num_envs

    def reset(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        for i, e in enumerate(self.envs):
            o, lm, cur = e.reset()
            self._cur_obs[i] = o
            self._cur_legal[i] = lm
            self._cur_player[i] = cur
            self._cur_hands[i] = e.get_player_hands()
        self._ep_score[:] = 0.0
        self._ep_len[:] = 0
        return (
            self._cur_obs.copy(),
            self._cur_legal.copy(),
            self._cur_player.copy(),
            self._cur_hands.copy(),
        )

    def step(self, actions: np.ndarray):
        actions = np.asarray(actions, dtype=np.int64).reshape(-1)
        rewards = np.zeros(self._num_envs, dtype=np.float32)
        dones = np.zeros(self._num_envs, dtype=bool)
        actors = self._cur_player.copy()

        completed: list[dict | None] = [None] * self._num_envs
        for i in range(self._num_envs):
            o, lm, cur, r, done, _info = self.envs[i].step(int(actions[i]))
            self._ep_score[i] += r
            self._ep_len[i] += 1
            rewards[i] = r
            dones[i] = done
            if done:
                completed[i] = {
                    "score": float(self._ep_score[i]),
                    "length": int(self._ep_len[i]),
                }
                self._ep_score[i] = 0.0
                self._ep_len[i] = 0
                o2, lm2, cur2 = self.envs[i].reset()
                self._cur_obs[i] = o2
                self._cur_legal[i] = lm2
                self._cur_player[i] = cur2
                self._cur_hands[i] = self.envs[i].get_player_hands()
            else:
                self._cur_obs[i] = o
                self._cur_legal[i] = lm
                self._cur_player[i] = cur
                self._cur_hands[i] = self.envs[i].get_player_hands()

        info = {
            "completed": completed,
            "actor": actors,
        }
        return (
            self._cur_obs.copy(),
            self._cur_legal.copy(),
            self._cur_player.copy(),
            self._cur_hands.copy(),
            rewards,
            dones,
            info,
        )

    def close(self):
        for e in self.envs:
            e.close()
