"""Unit tests for env construction and observation/action contracts.

These tests do **not** require CarRacing to be installed; they verify our
`make_env` factory and the public API of a non-vectorized CarRacing env.
If CarRacing is unavailable (missing swig/box2d), the CarRacing tests are
skipped so the rest of the suite still runs.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest


def _has_carracing() -> bool:
    return importlib.util.find_spec("gymnasium") is not None and importlib.util.find_spec(
        "box2d"
    ) is not None


skip_no_carracing = pytest.mark.skipif(
    not _has_carracing(),
    reason="gymnasium[box2d] not installed",
)


def test_imports():
    from drive_rl import env as env_mod
    from drive_rl import utils

    assert env_mod.ENV_ID == "CarRacing-v3"
    assert hasattr(utils, "set_global_seed")
    assert hasattr(utils, "select_device")


def test_set_global_seed_is_deterministic():
    from drive_rl.utils import set_global_seed

    set_global_seed(0)
    a = np.random.rand(5)
    set_global_seed(0)
    b = np.random.rand(5)
    np.testing.assert_allclose(a, b)


@skip_no_carracing
def test_make_env_shapes():
    from drive_rl.env import make_env

    env = make_env(n_envs=1, seed=0, n_stack=4, vec_cls=None)
    obs_space = env.observation_space
    act_space = env.action_space
    assert obs_space.shape == (4, 84, 84)
    assert obs_space.dtype == np.uint8
    assert act_space.shape == (3,)
    assert float(act_space.low[0]) == -1.0 and float(act_space.high[0]) == 1.0
    obs = env.reset()
    assert obs.shape == (1, 4, 84, 84)
    assert obs.dtype == np.uint8
    env.close()


@skip_no_carracing
def test_env_step_returns_correct_types():
    from drive_rl.env import make_env

    env = make_env(n_envs=1, seed=1, n_stack=4, vec_cls=None)
    obs = env.reset()
    action = env.action_space.sample()
    obs, reward, done, info = env.step([action])
    assert obs.shape == (1, 4, 84, 84)
    assert isinstance(float(reward[0]), float)
    assert isinstance(bool(done[0]), bool)
    assert isinstance(info, list)
    env.close()
