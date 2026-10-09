"""Tests for the racing environment.

Environment construction is slow (MetaDrive builds the map and the simulation
world), so the whole module is skipped when MetaDrive is unavailable. The pure
geometry and reward-arithmetic tests live in the other test modules and always
run.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

_HAS_METADRIVE = importlib.util.find_spec("metadrive") is not None

skip_no_metadrive = pytest.mark.skipif(
    not _HAS_METADRIVE, reason="metadrive-simulator not installed"
)


def test_camera_observation_layout():
    """(H, W, C, stack) must flatten to (stack * C, H, W) in the right order."""
    from drive_rl.racing_env import CameraObservation

    h, w, c, stack = 8, 6, 3, 4
    # Channel 0 of every stacked frame is filled with its stack index.
    frame = np.zeros((h, w, c, stack), dtype=np.float32)
    for s in range(stack):
        frame[..., 0, s] = s
    out = CameraObservation.to_channels_first({"image": frame, "state": np.zeros(3)})

    assert out.shape == (stack * c, h, w)
    assert out.dtype == np.float32
    # Output channel s * 3 holds stack index s.
    for s in range(stack):
        assert np.all(out[s * 3] == s)


def test_camera_observation_requires_image_key():
    from drive_rl.racing_env import CameraObservation

    with pytest.raises(KeyError):
        CameraObservation.to_channels_first({"state": np.zeros(3)})


def test_racing_params_penalties_are_scaled_to_step_progress():
    """Penalties must be comparable to per-step progress, not orders above it.

    A step covers roughly 2 m of a ~4500 m lap, so per-step progress is about
    progress_reward / 2270. A penalty in the single digits would swamp it.
    """
    from drive_rl.racing_env import RacingParams

    params = RacingParams()
    per_step_progress = params.progress_reward / 2270.0
    assert params.offroad_penalty < 100 * per_step_progress
    assert params.lateral_penalty * params.lateral_penalty_cap**2 < 100 * per_step_progress
    # A completed lap must still dominate, or laps would never be worth chasing.
    assert params.lap_bonus + params.progress_reward > params.offroad_penalty


@skip_no_metadrive
def test_env_builds_with_expected_spaces():
    from drive_rl.racing_env import make_racing_env

    env = make_racing_env("suzuka", obs_mode="state", log_level=50)
    try:
        env.reset(seed=0)
        assert isinstance(env.action_space.shape, tuple)
        assert env.action_space.shape == (2,)
        assert env.lap_length > 1000.0
        assert env.circuit.name == "suzuka"
    finally:
        env.close()


@skip_no_metadrive
def test_step_seconds_comes_from_config():
    """One env step is decision_repeat x physics_world_step_size."""
    from drive_rl.racing_env import make_racing_env

    env = make_racing_env("suzuka", log_level=50)
    try:
        expected = env.config["decision_repeat"] * env.config["physics_world_step_size"]
        assert env.step_seconds == pytest.approx(expected)
        # MetaDrive's defaults are 5 x 0.02 = 0.1 s.
        assert env.step_seconds == pytest.approx(0.1)
    finally:
        env.close()


@skip_no_metadrive
def test_reference_line_agrees_with_env_spawn():
    """The car must spawn on the racing line, or lap progress starts wrong."""
    from drive_rl.racing_env import make_racing_env

    env = make_racing_env("suzuka", log_level=50)
    try:
        _obs, info = env.reset(seed=0)
        assert abs(info["lateral"]) < 1.0
        assert info["on_lane"]
    finally:
        env.close()


@skip_no_metadrive
def test_progress_reward_matches_analytic_value():
    """Reward must equal progress_reward * (distance covered / lap length)."""
    from drive_rl.racing_env import RacingParams, make_racing_env

    params = RacingParams(offroad_penalty=0.0, lateral_penalty=0.0)
    env = make_racing_env("suzuka", racing=params, log_level=50)
    try:
        env.reset(seed=0)
        for _ in range(150):  # reach cruising speed down the straight
            _obs, reward, _term, _trunc, _info = env.step(np.array([0.0, 0.6], np.float32))
        rewards = []
        for _ in range(40):
            _obs, reward, _term, _trunc, info = env.step(np.array([0.0, 0.6], np.float32))
            rewards.append(reward)
        expected = params.progress_reward * info["progress"] / env.lap_length
        # Per-step progress is small; this is a sanity check on the formula.
        assert abs(float(np.mean(rewards)) - expected) < 1e-3
    finally:
        env.close()


@skip_no_metadrive
def test_unknown_circuit_raises():
    from drive_rl.racing_env import make_racing_env

    with pytest.raises(KeyError):
        make_racing_env("imola", log_level=50)


def test_invalid_obs_mode_raises():
    from drive_rl.racing_env import build_racing_config

    with pytest.raises(ValueError):
        build_racing_config("suzuka", obs_mode="lidar")
