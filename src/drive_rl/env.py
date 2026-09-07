"""Environment factory: build a wrapped, vectorized CarRacing-v3 env.

Pipeline applied to each env:
  1. `gymnasium.make("CarRacing-v3", continuous=True)` — continuous Box-3 action.
  2. `RecordEpisodeStatistics` — surface `episode` info for SB3 callbacks.
  3. Custom `GrayscaleResize` — RGB → single channel → 84×84 (keeps channel dim).
  4. SB3 `VecFrameStack(n_stack=4)` — stack last N frames into a single obs.

The resulting VecEnv observation space is `Box(0, 255, (4, 84, 84), uint8)`,
ready for SB3's `CnnPolicy`.
"""

from __future__ import annotations

from typing import Any

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import ObservationWrapper, spaces
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv, VecFrameStack

ENV_ID = "CarRacing-v3"
TARGET_SHAPE = (84, 84)


class GrayscaleResize(ObservationWrapper):
    """Convert RGB to grayscale and resize to 84x84, keeping channel dim."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        assert isinstance(env.observation_space, spaces.Box)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(TARGET_SHAPE[0], TARGET_SHAPE[1], 1), dtype=np.uint8
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        # obs: (H, W, 3) RGB uint8
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)  # (H, W)
        resized = cv2.resize(gray, TARGET_SHAPE[::-1], interpolation=cv2.INTER_AREA)  # (84, 84)
        return resized[:, :, None]  # (84, 84, 1)


def _make_thunk(seed: int, rank: int, render_mode: str | None = None) -> Any:
    """Return a zero-arg callable that builds one preprocessed CarRacing-v3 env."""
    def _thunk() -> gym.Env:
        env = gym.make(ENV_ID, continuous=True, render_mode=render_mode)
        env = GrayscaleResize(env)  # (84, 84, 1)
        env = Monitor(env)  # records episode reward/length for SB3 logging
        env.reset(seed=seed + rank)
        return env
    return _thunk


def make_env(
    n_envs: int = 4,
    seed: int = 0,
    n_stack: int = 4,
    vec_cls: type[VecEnv] | None = None,
    render_mode: str | None = None,
) -> VecEnv:
    """Construct a vectorized, frame-stacked CarRacing-v3 environment.

    Args:
        n_envs: Number of parallel envs. >1 enables `SubprocVecEnv` by default
            to keep training off the GIL; pass `vec_cls=DummyVecEnv` for debug.
        seed: Base seed; each sub-env gets `seed + rank`.
        n_stack: Number of consecutive frames stacked along the channel axis.
        vec_cls: Override the VecEnv class. Defaults to SubprocVecEnv when
            `n_envs > 1`, else DummyVecEnv.
        render_mode: `"human"` to display, `"rgb_array"` to capture, or `None`
            for headless training (recommended; "human" is single-env only).

    Returns:
        A VecEnv whose `observation_space` is `Box(0, 255, (n_stack, 84, 84), uint8)`.
    """
    if vec_cls is None:
        vec_cls = SubprocVecEnv if n_envs > 1 else DummyVecEnv

    env_fns = [_make_thunk(seed=seed, rank=i, render_mode=render_mode) for i in range(n_envs)]
    env = vec_cls(env_fns)
    env = VecFrameStack(env, n_stack=n_stack, channels_order="first")
    return env


def make_eval_env(seed: int = 0, n_stack: int = 4) -> VecEnv:
    """Single-env, deterministic VecEnv for evaluation and video recording."""
    return make_env(n_envs=1, seed=seed, n_stack=n_stack, vec_cls=DummyVecEnv)
