"""Multi-agent ViZDoom deathmatch with PettingZoo ParallelEnv API.

Each player runs inside its own OS process — ViZDoom's networked runtime is
not thread-safe, so same-process multiplayer segfaults once ticks advance.
Player 0 hosts; the rest join as clients on 127.0.0.1. The parent process
talks to each worker over a ``multiprocessing.Pipe`` and fan-outs step/reset
commands.

Uses ``ASYNC_PLAYER`` mode: in strict ``PLAYER`` (tick-sync) mode a host that
returns from ``advance_action`` first drives the opposite peer into a stuck
"still advancing" state — the sync barrier needs both sides inside
``advance_action`` simultaneously, and our pipe-driven loop can't guarantee
that. ASYNC_PLAYER ticks on its own clock and sidesteps the deadlock.

Defaults to ``multi_duel.cfg`` (1-vs-1 duel) — the official ViZDoom
multiplayer scenario. For 3+ players use ``multi.cfg`` or a custom ``.cfg`` /
``.wad`` combo. ``deathmatch.cfg`` is single-player only and will crash on
host/join.
"""
from __future__ import annotations

import multiprocessing as mp
import socket
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import vizdoom as vzd
from pettingzoo.utils.env import ParallelEnv

DEFAULT_BUTTONS: list[vzd.Button] = [
    vzd.Button.MOVE_FORWARD,
    vzd.Button.MOVE_BACKWARD,
    vzd.Button.TURN_LEFT,
    vzd.Button.TURN_RIGHT,
    vzd.Button.MOVE_LEFT,
    vzd.Button.MOVE_RIGHT,
    vzd.Button.ATTACK,
]

DEFAULT_VARIABLES: list[vzd.GameVariable] = [
    vzd.GameVariable.FRAGCOUNT,
    vzd.GameVariable.DEATHCOUNT,
    vzd.GameVariable.HEALTH,
    vzd.GameVariable.ARMOR,
    vzd.GameVariable.DAMAGECOUNT,         # damage you dealt (monotonic)
    vzd.GameVariable.DAMAGE_TAKEN,        # damage received (monotonic)
    vzd.GameVariable.HITCOUNT,            # hits landed
    vzd.GameVariable.HITS_TAKEN,          # hits received
    vzd.GameVariable.AMMO2,
    vzd.GameVariable.AMMO3,
    vzd.GameVariable.SELECTED_WEAPON,
    vzd.GameVariable.SELECTED_WEAPON_AMMO,
]

VAR_INDEX = {v: i for i, v in enumerate(DEFAULT_VARIABLES)}

# Sample-Factory-inspired shaping. Frag (+1) is the dominant terminal signal,
# but dense rewards (damage dealt, healing, pickups) provide useful gradient
# even when kills are rare.
REWARD_CONFIG: dict[str, float] = {
    "frag": 1.0,            # +1 per kill
    "death": -0.1,          # soft death penalty (was -0.5 — too risk-averse)
    "damage_dealt": 0.005,  # per HP of damage landed
    "health": 0.01,         # symmetric: + on heal pickup, - on taking damage
    "armor_gain": 0.005,    # only positive delta
    "ammo_gain": 0.002,     # only positive delta (sum across slots)
    "living": 0.0005,       # mild per-step bonus to discourage freezing
}


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _player_worker(
    conn: Connection,
    *,
    player_idx: int,
    is_host: bool,
    num_players: int,
    scenario: str,
    port: int,
    episode_timeout_seconds: float,
    frame_skip: int,
    frame_shape: tuple[int, int],
    ticrate: int,
    seed: int | None,
    bots_filled: int,
    record: bool = False,
) -> None:
    import vizdoom as vzd

    H, W = frame_shape
    game = vzd.DoomGame()
    try:
        game.load_config(scenario)
        game.set_window_visible(False)
        game.set_mode(vzd.Mode.ASYNC_PLAYER)
        game.set_ticrate(ticrate)
        game.set_screen_resolution(vzd.ScreenResolution.RES_320X240)
        game.set_available_buttons(DEFAULT_BUTTONS)
        game.set_available_game_variables(DEFAULT_VARIABLES)
        if seed is not None:
            game.set_seed(seed + player_idx)

        timelimit_min = max(episode_timeout_seconds / 60.0, 0.1)
        if is_host:
            args = (
                f"-host {num_players} -port {port} "
                f"-netmode 0 -deathmatch "
                f"+timelimit {timelimit_min:.3f} "
                f"+sv_forcerespawn 1 +sv_noautoaim 1 "
                f"+sv_noexit 1 +fraglimit 0 "
                f"+name Player{player_idx} +colorset {player_idx}"
            )
        else:
            args = (
                f"-join 127.0.0.1 -port {port} "
                f"+name Player{player_idx} +colorset {player_idx}"
            )
        game.add_game_args(args)

        game.init()
        if is_host:
            for _ in range(bots_filled):
                game.send_game_command("addbot")

        n_buttons = game.get_available_buttons_size()
        n_vars = game.get_available_game_variables_size()
    except BaseException as exc:  # noqa: BLE001
        conn.send(("init_error", repr(exc)))
        conn.close()
        return

    import cv2

    def _observe() -> dict[str, np.ndarray]:
        state = game.get_state()
        if state is None:
            return {
                "screen": np.zeros((3, H, W), dtype=np.uint8),
                "gamevars": np.zeros(n_vars, dtype=np.float32),
            }
        raw = state.screen_buffer
        resized = np.stack(
            [
                cv2.resize(
                    np.ascontiguousarray(raw[c]),
                    (W, H),
                    interpolation=cv2.INTER_AREA,
                )
                for c in range(3)
            ]
        )
        return {
            "screen": resized.astype(np.uint8),
            "gamevars": np.asarray(state.game_variables, dtype=np.float32),
        }

    def _raw_hwc_frame() -> np.ndarray:
        state = game.get_state()
        if state is None:
            return np.zeros((240, 320, 3), dtype=np.uint8)
        # CRCGCB (3, H, W) -> HWC RGB
        return np.ascontiguousarray(
            np.transpose(state.screen_buffer, (1, 2, 0))
        )

    prev = {
        "frags": 0, "deaths": 0, "health": 100, "armor": 0,
        "damage_dealt": 0, "ammo": 0,
    }
    conn.send(("ready", {"obs": _observe(), "n_buttons": n_buttons, "n_vars": n_vars}))

    def _read_var(var: vzd.GameVariable, fallback: int) -> int:
        state = game.get_state()
        if state is None:
            return fallback
        try:
            return int(state.game_variables[VAR_INDEX[var]])
        except (IndexError, KeyError):
            return fallback

    try:
        while True:
            cmd, arg = conn.recv()
            if cmd == "step":
                action_vec = [0] * n_buttons
                action_vec[int(arg) % n_buttons] = 1
                game.make_action(action_vec, frame_skip)

                # sv_forcerespawn=1 means the server respawns dead players
                # automatically. Explicit respawn_player() here crashes
                # ViZDoom when is_episode_finished is already True, so we
                # just report dead=True without intervening.
                dead = bool(game.is_player_dead())
                terminated = bool(game.is_episode_finished())

                # clamped reads; engine can momentarily report stale / huge
                # sentinels (e.g. HEALTH right after respawn).
                def _rd(gv, lo=0, hi=10_000_000):
                    try:
                        v = int(game.get_game_variable(gv))
                    except Exception:
                        return 0
                    return max(lo, min(hi, v))

                cur_frags = _rd(vzd.GameVariable.FRAGCOUNT, -100, 100)
                cur_deaths = _rd(vzd.GameVariable.DEATHCOUNT, 0, 1000)
                cur_health = _rd(vzd.GameVariable.HEALTH, 0, 200)
                cur_armor = _rd(vzd.GameVariable.ARMOR, 0, 200)
                cur_dmg_dealt = _rd(vzd.GameVariable.DAMAGECOUNT, 0, 1_000_000)
                cur_ammo = _rd(vzd.GameVariable.AMMO2, 0, 1000) + _rd(
                    vzd.GameVariable.AMMO3, 0, 1000,
                )

                frag_delta = max(0, min(5, cur_frags - prev["frags"]))
                death_delta = max(0, min(5, cur_deaths - prev["deaths"]))
                health_delta = max(-200, min(200, cur_health - prev["health"]))
                armor_delta = max(0, cur_armor - prev["armor"])
                dmg_dealt_delta = max(0, min(500, cur_dmg_dealt - prev["damage_dealt"]))
                ammo_delta = max(0, min(500, cur_ammo - prev["ammo"]))

                reward = (
                    REWARD_CONFIG["frag"] * frag_delta
                    + REWARD_CONFIG["death"] * death_delta
                    + REWARD_CONFIG["damage_dealt"] * dmg_dealt_delta
                    + REWARD_CONFIG["health"] * health_delta
                    + REWARD_CONFIG["armor_gain"] * armor_delta
                    + REWARD_CONFIG["ammo_gain"] * ammo_delta
                    + REWARD_CONFIG["living"]
                )
                prev["frags"] = cur_frags
                prev["deaths"] = cur_deaths
                prev["armor"] = cur_armor
                prev["health"] = cur_health
                prev["damage_dealt"] = cur_dmg_dealt
                prev["ammo"] = cur_ammo

                info = {
                    "frags": cur_frags,
                    "deaths": cur_deaths,
                    "health": cur_health,
                    "armor": cur_armor,
                    "damage_dealt": cur_dmg_dealt,
                    "ammo": cur_ammo,
                    "dead": dead,
                }
                if record:
                    info["frame"] = _raw_hwc_frame()
                conn.send(("step_ok", (_observe(), float(reward), terminated, False, info)))

            elif cmd == "close":
                break
            else:
                conn.send(("error", f"unknown cmd: {cmd!r}"))
    except (EOFError, BrokenPipeError):
        pass
    finally:
        try:
            game.close()
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass


class VizDoomMultiAgentEnv(ParallelEnv):
    """PettingZoo ParallelEnv: one DoomGame per subprocess, networked multiplayer."""

    metadata = {"name": "vizdoom_multi_v0", "render_modes": ["rgb_array"]}

    def __init__(
        self,
        num_players: int = 2,
        scenario: str | None = None,
        episode_timeout_seconds: float = 720.0,
        frame_skip: int = 4,
        frame_shape: tuple[int, int] = (84, 84),
        ticrate: int = 1000,
        port: int | None = None,
        render_mode: str | None = None,
        seed: int | None = None,
        bots_filled: int = 0,
        record: bool = False,
    ):
        if num_players < 2:
            raise ValueError("num_players must be >= 2")
        self.num_players = num_players
        self.scenario = scenario or str(
            Path(vzd.scenarios_path) / "multi_duel.cfg"
        )
        self.episode_timeout_seconds = float(episode_timeout_seconds)
        self.frame_skip = int(frame_skip)
        self.frame_shape = tuple(frame_shape)
        self.ticrate = int(ticrate)
        self._port_override = port
        self.render_mode = render_mode
        self.bots_filled = int(bots_filled)
        self.record = bool(record)
        self._seed = seed

        self.possible_agents: list[str] = [f"player_{i}" for i in range(num_players)]
        self.agents: list[str] = []
        self._pipes: dict[str, Connection] = {}
        self._procs: dict[str, mp.Process] = {}
        self._last_obs: dict[str, dict[str, np.ndarray]] = {}
        self._ctx = mp.get_context("spawn")

        H, W = self.frame_shape
        self.n_buttons = len(DEFAULT_BUTTONS)
        self.n_vars = len(DEFAULT_VARIABLES)
        self.action_spaces = {
            a: gym.spaces.Discrete(self.n_buttons) for a in self.possible_agents
        }
        self.observation_spaces = {
            a: gym.spaces.Dict(
                {
                    "screen": gym.spaces.Box(
                        low=0, high=255, shape=(3, H, W), dtype=np.uint8
                    ),
                    "gamevars": gym.spaces.Box(
                        low=-1e6,
                        high=1e6,
                        shape=(self.n_vars,),
                        dtype=np.float32,
                    ),
                }
            )
            for a in self.possible_agents
        }

    def observation_space(self, agent: str):
        return self.observation_spaces[agent]

    def action_space(self, agent: str):
        return self.action_spaces[agent]

    @property
    def num_agents(self) -> int:
        return len(self.possible_agents)

    @property
    def max_num_agents(self) -> int:
        return len(self.possible_agents)

    @property
    def num_envs(self) -> int:
        return 1

    @property
    def state_space(self) -> gym.spaces.Dict:
        """For independent critics (IPPO/self-play) we don't need a true
        global state — each agent's value function is per-agent. Use player_0's
        observation space as the placeholder so skrl's memory can allocate
        matching tensors.
        """
        return self.observation_spaces[self.possible_agents[0]]

    def state(self) -> dict[str, np.ndarray]:
        if self._last_obs and self.possible_agents[0] in self._last_obs:
            return self._last_obs[self.possible_agents[0]]
        H, W = self.frame_shape
        return {
            "screen": np.zeros((3, H, W), dtype=np.uint8),
            "gamevars": np.zeros(self.n_vars, dtype=np.float32),
        }

    def _spawn_all(self) -> None:
        port = self._port_override if self._port_override is not None else _free_port()
        for i, agent in enumerate(self.possible_agents):
            parent_conn, child_conn = self._ctx.Pipe(duplex=True)
            proc = self._ctx.Process(
                target=_player_worker,
                kwargs=dict(
                    conn=child_conn,
                    player_idx=i,
                    is_host=(i == 0),
                    num_players=self.num_players,
                    scenario=self.scenario,
                    port=port,
                    episode_timeout_seconds=self.episode_timeout_seconds,
                    frame_skip=self.frame_skip,
                    frame_shape=self.frame_shape,
                    ticrate=self.ticrate,
                    seed=self._seed,
                    bots_filled=self.bots_filled if i == 0 else 0,
                    record=self.record,
                ),
                daemon=True,
                name=f"vzd-{agent}",
            )
            proc.start()
            child_conn.close()
            self._pipes[agent] = parent_conn
            self._procs[agent] = proc

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, dict], dict[str, dict]]:
        if seed is not None:
            self._seed = seed
        self.close()
        self.agents = list(self.possible_agents)
        self._spawn_all()

        obs: dict[str, dict] = {}
        for agent, pipe in self._pipes.items():
            if not pipe.poll(timeout=90.0):
                self.close()
                raise TimeoutError(
                    f"{agent} worker failed to init within 90s"
                )
            tag, payload = pipe.recv()
            if tag == "init_error":
                self.close()
                raise RuntimeError(f"{agent} init failed: {payload}")
            if tag != "ready":
                self.close()
                raise RuntimeError(f"{agent} unexpected handshake: {tag}")
            obs[agent] = payload["obs"]
        self._last_obs = obs
        infos = {a: {} for a in self.agents}
        return obs, infos

    STEP_RECV_TIMEOUT = 30.0  # seconds; worker silent this long -> treat as hung

    def step(
        self, actions: dict[str, int]
    ) -> tuple[
        dict[str, dict],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict],
    ]:
        for agent in self.agents:
            self._pipes[agent].send(("step", int(actions[agent])))

        obs: dict[str, dict] = {}
        rewards: dict[str, float] = {}
        terms: dict[str, bool] = {}
        truncs: dict[str, bool] = {}
        infos: dict[str, dict] = {}

        for agent in self.agents:
            pipe = self._pipes[agent]
            if not pipe.poll(timeout=self.STEP_RECV_TIMEOUT):
                raise TimeoutError(
                    f"{agent} worker silent for >{self.STEP_RECV_TIMEOUT}s"
                )
            tag, payload = pipe.recv()
            if tag != "step_ok":
                raise RuntimeError(f"{agent} step failed: {tag} {payload!r}")
            o, r, term, trunc, info = payload
            obs[agent] = o
            rewards[agent] = float(r)
            terms[agent] = bool(term)
            truncs[agent] = bool(trunc)
            infos[agent] = info

        self._last_obs = obs
        if all(terms.values()):
            self.agents = []

        return obs, rewards, terms, truncs, infos

    def render(self) -> np.ndarray | None:
        if not self._last_obs:
            return None
        first = next(iter(self._last_obs.values()))
        return np.transpose(first["screen"], (1, 2, 0))

    def close(self) -> None:
        for pipe in self._pipes.values():
            try:
                pipe.send(("close", None))
            except (BrokenPipeError, EOFError, OSError):
                pass
        for pipe in self._pipes.values():
            try:
                pipe.close()
            except Exception:
                pass
        for proc in self._procs.values():
            proc.join(timeout=5.0)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=2.0)
        self._pipes = {}
        self._procs = {}
        self._last_obs = {}
        self.agents = []
