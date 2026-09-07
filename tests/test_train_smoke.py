"""Smoke test: run a 1k-step PPO training on CPU to verify the pipeline.

This is the heaviest test in the suite and is skipped if CarRacing isn't
installed. It exists so CI catches integration regressions early.
"""

from __future__ import annotations

import importlib.util

import pytest


def _has_carracing() -> bool:
    return importlib.util.find_spec("gymnasium") is not None and importlib.util.find_spec(
        "box2d"
    ) is not None


skip = pytest.mark.skipif(
    not _has_carracing(), reason="gymnasium[box2d] not installed"
)


@skip
def test_ppo_smoke_train_1k_steps(tmp_path):
    from drive_rl.env import make_env
    from stable_baselines3 import PPO

    env = make_env(n_envs=2, seed=0, n_stack=4, vec_cls=None)  # DummyVecEnv in test
    model = PPO("CnnPolicy", env, verbose=0, n_steps=128, batch_size=32, n_epochs=2)
    model.learn(total_timesteps=512)
    model.save(tmp_path / "smoke.zip")
    env.close()
    assert (tmp_path / "smoke.zip").is_file()
