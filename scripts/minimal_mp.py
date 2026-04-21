"""Bare-minimum ViZDoom multiplayer via multiprocessing — bypasses the env wrapper."""
from __future__ import annotations

import multiprocessing as mp
import sys
import traceback
from pathlib import Path


def worker(is_host: bool, port: int, num_players: int, timelimit_min: float = 0.2):
    import vizdoom as vzd

    try:
        g = vzd.DoomGame()
        g.load_config(str(Path(vzd.scenarios_path) / "multi_duel.cfg"))
        g.set_window_visible(False)
        g.set_mode(vzd.Mode.PLAYER)
        g.set_screen_resolution(vzd.ScreenResolution.RES_320X240)
        if is_host:
            g.add_game_args(
                f"-host {num_players} -port {port} -netmode 0 -deathmatch "
                f"+timelimit {timelimit_min} +sv_forcerespawn 1 +sv_noautoaim 1 "
                f"+name Host +colorset 0"
            )
        else:
            g.add_game_args(f"-join 127.0.0.1 -port {port} +name Guest +colorset 3")
        g.init()
        print(f"[{'HOST' if is_host else 'CLIENT'}] init ok", flush=True)

        n = g.get_available_buttons_size()
        steps = 0
        while not g.is_episode_finished():
            a = [0] * n
            a[steps % n] = 1
            g.make_action(a, 4)
            steps += 1
            if steps > 400:
                break
            if g.is_player_dead():
                g.respawn_player()
        print(f"[{'HOST' if is_host else 'CLIENT'}] done after {steps} steps", flush=True)
        g.close()
    except BaseException:
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


def main():
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    ctx = mp.get_context("spawn")
    procs = [
        ctx.Process(target=worker, args=(True, port, 2)),
        ctx.Process(target=worker, args=(False, port, 2)),
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=120)
    for p in procs:
        if p.is_alive():
            print(f"terminating {p.name}")
            p.terminate()
    for p in procs:
        print(f"{p.name} exit={p.exitcode}")


if __name__ == "__main__":
    main()
