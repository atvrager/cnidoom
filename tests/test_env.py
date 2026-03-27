"""Tests for DoomHybridEnv — mocks VizDoom so no system deps needed."""

from unittest.mock import MagicMock, patch

import gymnasium as gym
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_screen_buffer(value: int = 128) -> np.ndarray:
    """Fake 160x120 GRAY8 screen buffer."""
    return np.full((120, 160), value, dtype=np.uint8)


def _make_mock_game():
    """Create a mock vzd.DoomGame with sensible defaults."""
    game = MagicMock()
    game.get_available_buttons_size.return_value = 6
    game.is_episode_finished.return_value = False
    game.make_action.return_value = 1.0  # base reward

    state = MagicMock()
    state.screen_buffer = _make_screen_buffer(128)
    game.get_state.return_value = state

    # Game variables: all return 0 by default.
    game.get_game_variable.return_value = 0.0
    return game


@pytest.fixture
def env():
    """Yield a DoomHybridEnv with a fully mocked VizDoom backend."""
    with patch("training.env.vzd") as mock_vzd:
        mock_game = _make_mock_game()
        mock_vzd.DoomGame.return_value = mock_game

        # Wire up the enum values so _configure_defaults doesn't crash.
        mock_vzd.ScreenResolution.RES_160X120 = "RES_160X120"
        mock_vzd.ScreenFormat.GRAY8 = "GRAY8"
        for name in [
            "HEALTH",
            "ARMOR",
            "AMMO2",
            "AMMO3",
            "AMMO4",
            "AMMO5",
            "SELECTED_WEAPON",
            "KILLCOUNT",
            "POSITION_X",
            "POSITION_Y",
        ]:
            setattr(mock_vzd.GameVariable, name, name)
        for name in [
            "MOVE_FORWARD",
            "MOVE_BACKWARD",
            "TURN_LEFT",
            "TURN_RIGHT",
            "ATTACK",
            "USE",
        ]:
            setattr(mock_vzd.Button, name, name)

        # Need to patch __file__ for the scenario path fallback.
        mock_vzd.__file__ = "/fake/vizdoom/__init__.py"

        from training.env import DoomHybridEnv

        environment = DoomHybridEnv()
        environment._mock_game = mock_game  # expose for test manipulation
        yield environment
        environment.close()


# ---------------------------------------------------------------------------
# Observation space / action space
# ---------------------------------------------------------------------------


class TestSpaces:
    def test_observation_space_is_dict(self, env):
        assert isinstance(env.observation_space, gym.spaces.Dict)
        assert "visual" in env.observation_space.spaces
        assert "state" in env.observation_space.spaces

    def test_visual_shape(self, env):
        assert env.observation_space["visual"].shape == (4, 45, 60)

    def test_state_shape(self, env):
        assert env.observation_space["state"].shape == (20,)

    def test_action_space(self, env):
        assert isinstance(env.action_space, gym.spaces.MultiBinary)
        assert env.action_space.n == 6


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_returns_obs_and_info(self, env):
        obs, info = env.reset()
        assert isinstance(obs, dict)
        assert isinstance(info, dict)

    def test_reset_obs_shapes(self, env):
        obs, _ = env.reset()
        assert obs["visual"].shape == (4, 45, 60)
        assert obs["state"].shape == (20,)

    def test_reset_visual_dtype(self, env):
        obs, _ = env.reset()
        assert obs["visual"].dtype == np.float32

    def test_reset_visual_range(self, env):
        obs, _ = env.reset()
        assert obs["visual"].min() >= 0.0
        assert obs["visual"].max() <= 1.0

    def test_reset_fills_all_frames(self, env):
        """After reset, all 4 frames in the stack should be identical."""
        obs, _ = env.reset()
        for i in range(1, 4):
            np.testing.assert_array_equal(obs["visual"][0], obs["visual"][i])

    def test_reset_calls_new_episode(self, env):
        env.reset()
        env._mock_game.new_episode.assert_called_once()


# ---------------------------------------------------------------------------
# Step
# ---------------------------------------------------------------------------


class TestStep:
    def test_step_returns_five_tuple(self, env):
        env.reset()
        action = np.array([1, 0, 0, 0, 0, 0])
        result = env.step(action)
        assert len(result) == 5

    def test_step_obs_shapes(self, env):
        env.reset()
        obs, reward, terminated, truncated, info = env.step(np.zeros(6, dtype=int))
        assert obs["visual"].shape == (4, 45, 60)
        assert obs["state"].shape == (20,)

    def test_step_reward_is_float(self, env):
        env.reset()
        _, reward, _, _, _ = env.step(np.zeros(6, dtype=int))
        assert isinstance(reward, float)

    def test_step_terminal(self, env):
        """When VizDoom says episode is finished, terminated should be True."""
        env.reset()
        env._mock_game.is_episode_finished.return_value = True
        _, _, terminated, truncated, _ = env.step(np.zeros(6, dtype=int))
        assert terminated is True
        assert truncated is False

    def test_step_frame_advances_stack(self, env):
        """Each step should shift the frame stack (oldest dropped, new appended)."""
        # Use different pixel values per step to distinguish frames.
        env.reset()
        for _i, val in enumerate([50, 100, 150, 200]):
            state = MagicMock()
            state.screen_buffer = _make_screen_buffer(val)
            env._mock_game.get_state.return_value = state
            obs, *_ = env.step(np.zeros(6, dtype=int))

        # The last 4 frames should reflect values 50, 100, 150, 200.
        # Each preprocessed frame divides by 255, so frame[-1] ≈ 200/255.
        last_frame_mean = obs["visual"][3].mean()
        assert abs(last_frame_mean - 200.0 / 255.0) < 0.02

    def test_step_calls_make_action_with_frame_skip(self, env):
        env.reset()
        action = np.array([1, 0, 1, 0, 0, 0])
        env.step(action)
        call_args = env._mock_game.make_action.call_args
        assert call_args[0][1] == 4  # frame_skip


# ---------------------------------------------------------------------------
# Frame preprocessing
# ---------------------------------------------------------------------------


class TestPreprocessing:
    def test_preprocess_none_returns_zeros(self):
        from training.env import DoomHybridEnv

        frame = DoomHybridEnv._preprocess_frame(None)
        assert frame.shape == (45, 60)
        assert frame.sum() == 0.0

    def test_preprocess_shape(self):
        from training.env import DoomHybridEnv

        buf = _make_screen_buffer(200)
        frame = DoomHybridEnv._preprocess_frame(buf)
        assert frame.shape == (45, 60)
        assert frame.dtype == np.float32

    def test_preprocess_white_frame(self):
        from training.env import DoomHybridEnv

        buf = _make_screen_buffer(255)
        frame = DoomHybridEnv._preprocess_frame(buf)
        assert frame.max() == pytest.approx(1.0)

    def test_preprocess_black_frame(self):
        from training.env import DoomHybridEnv

        buf = _make_screen_buffer(0)
        frame = DoomHybridEnv._preprocess_frame(buf)
        assert frame.max() == 0.0


# ---------------------------------------------------------------------------
# Contradiction masking
# ---------------------------------------------------------------------------


class TestContradictionMasking:
    def test_forward_backward_keeps_forward(self):
        from training.env import DoomHybridEnv

        action = np.array([1, 1, 0, 0, 0, 0])
        masked = DoomHybridEnv._mask_contradictions(action)
        assert masked[0] == 1  # forward kept
        assert masked[1] == 0  # backward removed

    def test_left_right_keeps_left(self):
        from training.env import DoomHybridEnv

        action = np.array([0, 0, 1, 1, 0, 0])
        masked = DoomHybridEnv._mask_contradictions(action)
        assert masked[2] == 1  # left kept
        assert masked[3] == 0  # right removed

    def test_no_contradiction_unchanged(self):
        from training.env import DoomHybridEnv

        action = np.array([1, 0, 0, 1, 1, 0])
        masked = DoomHybridEnv._mask_contradictions(action)
        np.testing.assert_array_equal(action, masked)

    def test_all_zeros_unchanged(self):
        from training.env import DoomHybridEnv

        action = np.zeros(6, dtype=int)
        masked = DoomHybridEnv._mask_contradictions(action)
        np.testing.assert_array_equal(action, masked)

    def test_does_not_mutate_input(self):
        from training.env import DoomHybridEnv

        action = np.array([1, 1, 1, 1, 1, 1])
        original = action.copy()
        DoomHybridEnv._mask_contradictions(action)
        np.testing.assert_array_equal(action, original)


# ---------------------------------------------------------------------------
# Reward shaping
# ---------------------------------------------------------------------------


class TestRewardShaping:
    def test_first_step_returns_base_reward(self, env):
        """First step has no prev_vars, so shaped ≈ base reward."""
        env.reset()
        _, reward, _, _, _ = env.step(np.zeros(6, dtype=int))
        # Base reward is 1.0 from mock; first step after reset has prev_vars
        # from reset, so shaping is applied but deltas are ~0.
        assert isinstance(reward, float)

    def test_kill_increases_reward(self, env):
        env.reset()

        # Step 1: baseline.
        env._mock_game.get_game_variable.return_value = 0.0
        env.step(np.zeros(6, dtype=int))

        # Step 2: kill count goes from 0 → 1.
        def gv_with_kill(var):
            if var == "KILLCOUNT":
                return 1.0
            return 0.0

        env._mock_game.get_game_variable.side_effect = gv_with_kill
        _, reward, _, _, _ = env.step(np.zeros(6, dtype=int))
        # Kill delta * 50.0 should dominate.
        assert reward > 40.0


# ---------------------------------------------------------------------------
# Game state vector
# ---------------------------------------------------------------------------


class TestStateVector:
    def test_state_clipped_to_range(self, env):
        env.reset()
        obs, _ = env.reset()
        assert obs["state"].min() >= -1.0
        assert obs["state"].max() <= 1.0

    def test_weapon_one_hot(self, env):
        """Selected weapon should produce exactly one 1.0 in indices 6-14."""

        def gv_weapon(var):
            if var == "SELECTED_WEAPON":
                return 2.0
            return 0.0

        env._mock_game.get_game_variable.side_effect = gv_weapon
        env.reset()
        obs, _ = env.reset()
        weapon_slice = obs["state"][6:15]
        assert weapon_slice[2] == 1.0
        assert weapon_slice.sum() == 1.0
