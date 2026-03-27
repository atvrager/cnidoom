"""VizDoom Gymnasium environment with hybrid observations and reward shaping."""

from collections import deque
from pathlib import Path

import gymnasium as gym
import numpy as np
import vizdoom as vzd


class DoomHybridEnv(gym.Env):
    """Hybrid observation: downsampled frame stack + game state vector.

    Observations:
        visual: (4, 45, 60) float32 — 4 grayscale frames, channels-first
        state:  (20,) float32 — normalized game variables

    Actions:
        MultiBinary(6) — forward, backward, turn_l, turn_r, fire, use
        Contradictions (fwd+bwd, left+right) are masked after sampling.
    """

    metadata = {"render_modes": ["rgb_array"]}

    # Indices into the action vector.
    _ACT_FORWARD = 0
    _ACT_BACKWARD = 1
    _ACT_TURN_L = 2
    _ACT_TURN_R = 3

    def __init__(
        self,
        cfg_path: str | Path | None = None,
        frame_skip: int = 4,
        stack_size: int = 4,
        render_mode: str | None = None,
        obs_h: int = 45,
        obs_w: int = 60,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.frame_skip = frame_skip
        self.stack_size = stack_size
        self.obs_h = obs_h
        self.obs_w = obs_w

        self.game = vzd.DoomGame()
        if cfg_path is not None:
            self.game.load_config(str(cfg_path))
            self._resolve_scenario_wad(cfg_path)
        else:
            self._configure_defaults()
        self.game.set_window_visible(False)
        self.game.init()

        self.frames: deque[np.ndarray] = deque(maxlen=stack_size)

        n_buttons = self.game.get_available_buttons_size()
        self.action_space = gym.spaces.MultiBinary(n_buttons)

        self.observation_space = gym.spaces.Dict(
            {
                "visual": gym.spaces.Box(
                    0.0, 1.0, (stack_size, self.obs_h, self.obs_w), dtype=np.float32
                ),
                "state": gym.spaces.Box(-1.0, 1.0, (20,), dtype=np.float32),
            }
        )

        # Reward shaping bookkeeping.
        self._prev_vars: dict[str, float] = {}

    # ------------------------------------------------------------------
    # WAD resolution
    # ------------------------------------------------------------------

    def _resolve_scenario_wad(self, cfg_path: str | Path) -> None:
        """Resolve the scenario WAD path from VizDoom's bundled scenarios.

        VizDoom resolves doom_scenario_path relative to the .cfg file's
        directory. If the WAD doesn't exist there, try VizDoom's bundled
        scenarios directory as a fallback.
        """
        cfg_dir = Path(cfg_path).resolve().parent
        wad_name = Path(self.game.get_doom_scenario_path()).name
        local_wad = cfg_dir / wad_name
        if local_wad.exists():
            return
        bundled_wad = Path(vzd.__file__).parent / "scenarios" / wad_name
        if bundled_wad.exists():
            self.game.set_doom_scenario_path(str(bundled_wad))

    # ------------------------------------------------------------------
    # Default VizDoom configuration (used when no .cfg file is provided)
    # ------------------------------------------------------------------

    def _configure_defaults(self) -> None:
        """Minimal E1M1 config matching the project spec."""
        self.game.set_doom_scenario_path(
            str(Path(vzd.__file__).parent / "scenarios" / "basic.wad")
        )
        self.game.set_doom_map("map01")

        self.game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
        self.game.set_screen_format(vzd.ScreenFormat.GRAY8)

        for var in [
            vzd.GameVariable.HEALTH,
            vzd.GameVariable.ARMOR,
            vzd.GameVariable.AMMO2,
            vzd.GameVariable.AMMO3,
            vzd.GameVariable.AMMO4,
            vzd.GameVariable.AMMO5,
            vzd.GameVariable.SELECTED_WEAPON,
            vzd.GameVariable.KILLCOUNT,
            vzd.GameVariable.POSITION_X,
            vzd.GameVariable.POSITION_Y,
        ]:
            self.game.add_available_game_variable(var)

        for btn in [
            vzd.Button.MOVE_FORWARD,
            vzd.Button.MOVE_BACKWARD,
            vzd.Button.TURN_LEFT,
            vzd.Button.TURN_RIGHT,
            vzd.Button.ATTACK,
            vzd.Button.USE,
        ]:
            self.game.add_available_button(btn)

        self.game.set_episode_timeout(4200)  # ~2 min at 35 tics/sec
        self.game.set_living_reward(0.0)
        self.game.set_death_penalty(100.0)

    # ------------------------------------------------------------------
    # Frame preprocessing
    # ------------------------------------------------------------------

    def _preprocess_frame(self, buf: np.ndarray | None) -> np.ndarray:
        """160x120 GRAY8 → (obs_h, obs_w) float32 [0, 1]."""
        if buf is None:
            return np.zeros((self.obs_h, self.obs_w), dtype=np.float32)
        # Area downsample: take every other pixel in both dims → ~80x60,
        # then crop to obs_h rows, obs_w cols.
        frame = buf[::2, ::2].astype(np.float32) / 255.0
        return frame[: self.obs_h, : self.obs_w]

    def _stacked_frames(self) -> np.ndarray:
        """Return (stack_size, obs_h, obs_w) float32 array."""
        return np.array(self.frames, dtype=np.float32)

    # ------------------------------------------------------------------
    # Game state vector
    # ------------------------------------------------------------------

    def _get_state_vec(self) -> np.ndarray:
        """Extract and normalize a 20-float state vector from VizDoom."""
        gv = self.game.get_game_variable
        v = np.zeros(20, dtype=np.float32)

        v[0] = gv(vzd.GameVariable.HEALTH) / 200.0
        v[1] = gv(vzd.GameVariable.ARMOR) / 200.0
        v[2] = gv(vzd.GameVariable.AMMO2) / 50.0
        v[3] = gv(vzd.GameVariable.AMMO3) / 50.0
        v[4] = gv(vzd.GameVariable.AMMO4) / 50.0
        v[5] = gv(vzd.GameVariable.AMMO5) / 300.0

        sel = int(gv(vzd.GameVariable.SELECTED_WEAPON))
        if 0 <= sel < 9:
            v[6 + sel] = 1.0  # one-hot weapon (indices 6–14)

        # Velocity from position deltas — VizDoom doesn't expose momx/momy
        # directly in all configs, so we use position delta as a proxy.
        # Indices 15-16 reserved for velocity (filled by reward shaping diff).
        # Indices 17-19 reserved for future extensions.

        return np.clip(v, -1.0, 1.0)

    # ------------------------------------------------------------------
    # Reward shaping
    # ------------------------------------------------------------------

    def _shaped_reward(self, base_reward: float) -> float:
        """Add dense reward signals on top of VizDoom's sparse reward."""
        gv = self.game.get_game_variable

        curr = {
            "kills": gv(vzd.GameVariable.KILLCOUNT),
            "health": gv(vzd.GameVariable.HEALTH),
            "ammo2": gv(vzd.GameVariable.AMMO2),
            "pos_x": gv(vzd.GameVariable.POSITION_X),
            "pos_y": gv(vzd.GameVariable.POSITION_Y),
        }

        shaped = base_reward
        if self._prev_vars:
            kill_delta = curr["kills"] - self._prev_vars["kills"]
            health_delta = curr["health"] - self._prev_vars["health"]
            ammo_delta = curr["ammo2"] - self._prev_vars["ammo2"]

            dx = curr["pos_x"] - self._prev_vars["pos_x"]
            dy = curr["pos_y"] - self._prev_vars["pos_y"]
            movement = (dx**2 + dy**2) ** 0.5

            shaped += kill_delta * 50.0
            shaped += max(health_delta, 0.0) * 0.5  # only reward pickups
            shaped += max(ammo_delta, 0.0) * 0.2
            shaped += movement * 0.01
            shaped -= 0.001  # tiny time penalty

        self._prev_vars = curr
        return shaped

    # ------------------------------------------------------------------
    # Contradiction masking
    # ------------------------------------------------------------------

    @staticmethod
    def _mask_contradictions(action: np.ndarray) -> np.ndarray:
        """Zero out contradictory simultaneous actions.

        If both forward+backward are active, keep forward.
        If both turn_left+turn_right are active, keep turn_left.
        """
        a = action.copy()
        if a[DoomHybridEnv._ACT_FORWARD] and a[DoomHybridEnv._ACT_BACKWARD]:
            a[DoomHybridEnv._ACT_BACKWARD] = 0
        if a[DoomHybridEnv._ACT_TURN_L] and a[DoomHybridEnv._ACT_TURN_R]:
            a[DoomHybridEnv._ACT_TURN_R] = 0
        return a

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def step(self, action: np.ndarray):
        action = self._mask_contradictions(action)
        base_reward = self.game.make_action(action.tolist(), self.frame_skip)
        done = self.game.is_episode_finished()

        if not done:
            state = self.game.get_state()
            frame = self._preprocess_frame(state.screen_buffer)
            reward = self._shaped_reward(base_reward)
        else:
            frame = np.zeros((self.obs_h, self.obs_w), dtype=np.float32)
            reward = base_reward  # no shaping on terminal step

        self.frames.append(frame)
        obs = {"visual": self._stacked_frames(), "state": self._get_state_vec()}
        return obs, reward, done, False, {}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.game.new_episode()
        self._prev_vars = {}

        # Fill frame stack with the initial frame.
        state = self.game.get_state()
        frame = self._preprocess_frame(state.screen_buffer)
        self.frames.clear()
        for _ in range(self.stack_size):
            self.frames.append(frame)

        obs = {"visual": self._stacked_frames(), "state": self._get_state_vec()}
        return obs, {}

    def render(self):
        if self.render_mode == "rgb_array" and not self.game.is_episode_finished():
            state = self.game.get_state()
            if state is not None:
                return state.screen_buffer
        return None

    def close(self):
        self.game.close()
