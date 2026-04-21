"""Single-player ViZDoom smoke test — verifies install before multi-agent code runs."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import vizdoom as vzd


def main() -> None:
    game = vzd.DoomGame()
    game.load_config(str(Path(vzd.scenarios_path) / "basic.cfg"))
    game.set_window_visible(False)
    game.set_screen_resolution(vzd.ScreenResolution.RES_320X240)
    game.init()
    print(f"ViZDoom {vzd.__version__} initialized")
    print(f"  scenarios_path: {vzd.scenarios_path}")
    print(
        f"  screen: {game.get_screen_height()}x{game.get_screen_width()} "
        f"channels={game.get_screen_channels()}"
    )

    rng = np.random.default_rng(0)
    n_actions = game.get_available_buttons_size()
    game.new_episode()

    total_reward = 0.0
    steps = 0
    while not game.is_episode_finished():
        a = [0] * n_actions
        a[int(rng.integers(0, n_actions))] = 1
        total_reward += float(game.make_action(a, 4))
        steps += 1

    print(f"episode finished: steps={steps} total_reward={total_reward:.2f}")
    game.close()


if __name__ == "__main__":
    main()
