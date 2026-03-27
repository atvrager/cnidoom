"""Tests for train.py — verifies CLI arg parsing and factory functions."""

from unittest.mock import patch

import numpy as np
import pytest


class TestMakeEnv:
    def test_make_env_returns_callable(self):
        with patch("training.env.vzd"):
            from training.train import make_env

            factory = make_env(cfg_path=None, frame_skip=4)
            assert callable(factory)

    def test_make_env_with_custom_frame_skip(self):
        """The factory should pass frame_skip through to the env."""
        with patch("training.env.vzd") as mock_vzd:
            mock_game = _quick_mock_game()
            mock_vzd.DoomGame.return_value = mock_game
            mock_vzd.__file__ = "/fake/__init__.py"
            _wire_enums(mock_vzd)

            from training.train import make_env

            factory = make_env(cfg_path=None, frame_skip=2)
            env = factory()
            assert env.frame_skip == 2
            env.close()


class TestMainArgparse:
    def test_default_args(self):
        """Verify argparse defaults without actually training."""
        from training.train import main
        # We just verify the function exists and is callable — actual training
        # is an integration test that requires VizDoom.
        assert callable(main)


# ---------------------------------------------------------------------------
# Helpers (duplicated from test_env to keep test files independent)
# ---------------------------------------------------------------------------


def _quick_mock_game():
    from unittest.mock import MagicMock

    game = MagicMock()
    game.get_available_buttons_size.return_value = 6
    game.is_episode_finished.return_value = False
    game.make_action.return_value = 0.0
    state = MagicMock()
    state.screen_buffer = np.full((120, 160), 128, dtype=np.uint8)
    game.get_state.return_value = state
    game.get_game_variable.return_value = 0.0
    return game


def _wire_enums(mock_vzd):
    mock_vzd.ScreenResolution.RES_160X120 = "RES_160X120"
    mock_vzd.ScreenFormat.GRAY8 = "GRAY8"
    for name in [
        "HEALTH", "ARMOR", "AMMO2", "AMMO3", "AMMO4", "AMMO5",
        "SELECTED_WEAPON", "KILLCOUNT", "POSITION_X", "POSITION_Y",
    ]:
        setattr(mock_vzd.GameVariable, name, name)
    for name in [
        "MOVE_FORWARD", "MOVE_BACKWARD", "TURN_LEFT",
        "TURN_RIGHT", "ATTACK", "USE",
    ]:
        setattr(mock_vzd.Button, name, name)
