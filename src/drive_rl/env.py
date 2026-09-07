"""Environment factory: build a wrapped, vectorized CarRacing-v3 env.

Pipeline applied to each env:
  1. `gymnasium.make("CarRacing-v3", continuous=True)` — continuous Box-3 action.
  2. `RecordEpisodeStatistics` — surface `episode` info for SB3 callbacks.
  3. `GrayscaleObservation` — RGB → single channel.
  4. `ResizeObservation` — downsample to 84×84.
  5. SB3 `VecFrameStack(n_stack=4)` — stack last N frames into a single obs.

The resulting VecEnv observation space is `Box(0, 255, (4, 84, 84), uint8)`,
ready for SB3's `CnnPolicy`.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv, VecFrameStack

ENV_ID = "CarRacing-v3"


def _make_thunk(seed: int, rank: int, render_mode: str | None = None) -> Any:
    """Return a zero-arg callable that builds one preprocessed CarRacing-v3 env."""
    def _thunk() -> gym.Env:
        env = gym.make(ENV_ID, continuous=True, render_mode=render_mode)
        env = GrayscaleObservation(env, keep_dim=True)  # (96, 96, 1)
        env = ResizeObservation(env, shape=(84, 84))  # (84, 84, 1)
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
