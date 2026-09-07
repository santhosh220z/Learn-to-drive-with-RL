"""Tests for opt-in wrappers. Uses a stub env so we don't need box2d."""

from __future__ import annotations

import numpy as np
import pytest
from gymnasium import Env, spaces


class _StubBoxEnv(Env):
    """Minimal Box-action env exposing `linearVelocity`-like attribute."""

    observation_space = spaces.Box(0, 255, (4, 84, 84), dtype=np.uint8)
    action_space = spaces.Box(low=-np.ones(3, dtype=np.float32), high=np.ones(3, dtype=np.float32))

    def __init__(self) -> None:
        self._steps = 0

    def reset(self, *, seed=None, options=None):
        self._steps = 0
        return np.zeros(self.observation_space.shape, dtype=np.uint8), {}

    def step(self, action):
        self._steps += 1
        terminated = self._steps >= 5
        return (
            np.zeros(self.observation_space.shape, dtype=np.uint8),
            float(action[0]),
            terminated,
            False,
            {},
        )


def test_action_smoothing_passthrough_at_alpha_one():
    from drive_rl.wrappers import ActionSmoothingWrapper

    env = ActionSmoothingWrapper(_StubBoxEnv(), alpha=1.0)
    env.reset()
    obs, r, term, trunc, info = env.step(np.array([0.7, 0.2, 0.3], dtype=np.float32))
    # Alpha 1.0 means no smoothing: action passed through.
    assert r == pytest.approx(0.7, abs=1e-5)


def test_action_smoothing_smoothing_factor():
    from drive_rl.wrappers import ActionSmoothingWrapper

    env = ActionSmoothingWrapper(_StubBoxEnv(), alpha=0.5)
    env.reset()
    # First action: prev=0, alpha=0.5 -> smoothed = 0.5*action
    obs, r, *_ = env.step(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    assert r == pytest.approx(0.5, abs=1e-5)
    # Second action with same value: smoothed = 0.5*prev + 0.5*1.0 = 0.5*0.5 + 0.5 = 0.75
    obs, r, *_ = env.step(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    assert r == pytest.approx(0.75, abs=1e-5)


def test_action_smoothing_rejects_bad_alpha():
    from drive_rl.wrappers import ActionSmoothingWrapper

    with pytest.raises(ValueError):
        ActionSmoothingWrapper(_StubBoxEnv(), alpha=0.0)
    with pytest.raises(ValueError):
        ActionSmoothingWrapper(_StubBoxEnv(), alpha=1.5)


def test_reward_shaping_off_passthrough():
    from drive_rl.wrappers import RewardShapingWrapper

    env = RewardShapingWrapper(_StubBoxEnv())
    env.reset()
    _, r, _, _, _ = env.step(np.array([0.4, 0.0, 0.0], dtype=np.float32))
    assert r == pytest.approx(0.4, abs=1e-6)


def test_reward_shaping_steer_penalty():
    from drive_rl.wrappers import RewardShapingWrapper

    env = RewardShapingWrapper(_StubBoxEnv(), steer_penalty=1.0)
    env.reset()
    # First step: prev=0, steer change = |0.7-0| = 0.7
    _, r, _, _, _ = env.step(np.array([0.7, 0.0, 0.0], dtype=np.float32))
    assert r == pytest.approx(0.7 - 0.7, abs=1e-6)
    # Second step with same steer: change = 0
    _, r, _, _, _ = env.step(np.array([0.7, 0.0, 0.0], dtype=np.float32))
    assert r == pytest.approx(0.7, abs=1e-6)


def test_frame_skip_sums_rewards():
    from drive_rl.wrappers import FrameSkipWrapper

    env = FrameSkipWrapper(_StubBoxEnv(), skip=3)
    env.reset()
    obs, r, term, trunc, info = env.step(np.array([0.2, 0.0, 0.0], dtype=np.float32))
    # Three steps each contributing 0.2 -> 0.6, then terminated at step 5 after 3 skips
    assert r == pytest.approx(0.6, abs=1e-6)
