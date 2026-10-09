"""Racing environment built on MetaDrive, with lap tracking and a racing reward.

MetaDrive ships an urban-driving task: get from A to B without crashing. That is
not racing. This module turns it into a racing task by layering on three things
MetaDrive does not provide.

**A real track.** :mod:`drive_rl.circuits` describes a circuit as a chain of
straight and arc blocks, and ``build_map_class`` turns one into a MetaDrive map
class, registered through a map manager that loads it once per episode.

**A racing reward.** MetaDrive's native reward pays for travelling forwards in
lane. Here reward is driven by *progress around the lap*, measured against a
:class:`~drive_rl.circuits.ReferenceLine` built from the same circuit geometry.
Progress is therefore well defined across block boundaries and does not depend on
MetaDrive's internal lane or node identifiers.

**Lap counting.** Crossing the start/finish line is detected by arc length
wrapping from the end of the lap back to zero.

Timing note: MetaDrive advances physics by ``physics_world_step_size`` (0.02 s)
per environment step, i.e. 50 Hz control. A lap of Suzuka is roughly 4000 steps,
so lap bonuses are sparse relative to a PPO rollout. The dense progress term --
worth ``progress_reward`` across a full lap -- is what actually carries the
learning signal, and is why training does not depend on lap boundaries lining up
with rollouts.

Two observation modes are supported, because they suit different purposes:

``obs_mode="state"``
    MetaDrive's vector observation (lidar plus ego state). Trains quickly on CPU,
    and is the mode MetaDrive's own PPO example uses.

``obs_mode="camera"``
    A rendered first-person RGB frame, flattened to channels-first
    ``(stack * 3, H, W)`` for Stable-Baselines3's ``CnnPolicy``. Closer to how a
    driver sees the track, and far more expensive.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from drive_rl.circuits import Circuit, ReferenceLine, build_map_class, get_circuit, map_config

DEFAULT_CIRCUIT = "suzuka"
DEFAULT_IMAGE_SIZE = 84
DEFAULT_STACK_SIZE = 3

# Reward weights, in lap units: a clean lap earns `progress_reward`.
RACING_CONFIG: dict[str, Any] = {
    "use_render": False,
    "manual_control": False,
    # A circuit is a closed loop with no traffic and nowhere else to go.
    "traffic_density": 0.0,
    "num_scenarios": 1,
    "log_level": 20,
    # No rendered images unless the caller asks for them.
    "image_observation": False,
    "norm_pixel": True,
    "stack_size": DEFAULT_STACK_SIZE,
    # Sensing. Lidar reads the road ahead; the side detector reads the track
    # edges, which is what tells a driver they are running out of road.
    "vehicle_config": {
        "enable_reverse": False,
        "show_navi_mark": False,
        "show_navigation_arrow": False,
        "show_line_to_dest": False,
        "lidar": {
            "num_lasers": 72,
            "distance": 50,
            "num_others": 0,
            "gaussian_noise": 0.0,
            "dropout_prob": 0.0,
        },
        "side_detector": {
            "num_lasers": 72,
            "distance": 50,
            "gaussian_noise": 0.0,
        },
        "lane_line_detector": {"num_lasers": 0, "distance": 20},
    },
    # Termination belongs to this module, not to MetaDrive's urban rules.
    "out_of_road_done": False,
    "crash_vehicle_done": False,
    "crash_object_done": False,
    "out_of_route_done": False,
    "on_broken_line_done": False,
    "on_continuous_line_done": False,
    "horizon": None,
}


@dataclass(frozen=True)
class RacingParams:
    """Reward and termination knobs owned by this module.

    These are deliberately kept out of MetaDrive's config: MetaDrive validates
    its config keys strictly and would reject unknown ones. They are attached to
    the environment as ``env.racing`` instead.

    Reward is expressed in *lap units*: a clean lap earns ``progress_reward``,
    prorated across the steps that cover it.

    That choice has a consequence worth stating plainly, because getting it wrong
    silently ruins training. A step covers roughly 2 m of a 4545 m lap, so the
    progress reward per step is only ``progress_reward / 2270`` -- about 0.004 at
    the default of 10. Penalties are charged *per step*, so they have to be
    scaled to that same order of magnitude. An off-road penalty in the single
    digits would be a thousand times the progress signal and would dominate every
    gradient, so the defaults below sit around 0.1.

    Attributes:
        progress_reward: Awarded for a full lap, prorated every step.
        lateral_penalty: Charged on the squared lateral offset in track-widths.
            Capped at ``lateral_penalty_cap`` track-widths: once the car is
            properly off the road the off-road penalty is the meaningful signal,
            and an uncapped quadratic here would dwarf the progress reward.
        lateral_penalty_cap: Maximum offset, in track-half-widths, that the
            lateral penalty responds to.
        offroad_penalty: Charged each step spent off the racing surface.
        lap_bonus: Awarded for crossing the start/finish line.
        offroad_tolerance: Consecutive off-track steps tolerated before the
            episode ends.
        stuck_tolerance: Consecutive near-stationary steps tolerated before the
            episode ends.
        stuck_speed_km_h: Speed below which the car counts as stuck.
        max_episode_seconds: Episode time cap, in simulated seconds.
    """

    progress_reward: float = 10.0
    lateral_penalty: float = 0.02
    lateral_penalty_cap: float = 1.5
    offroad_penalty: float = 0.2
    lap_bonus: float = 5.0
    offroad_tolerance: int = 30
    stuck_tolerance: int = 250
    stuck_speed_km_h: float = 5.0
    max_episode_seconds: float = 120.0


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``."""
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def build_racing_config(
    circuit: str | Circuit = DEFAULT_CIRCUIT,
    *,
    obs_mode: str = "state",
    image_size: int = DEFAULT_IMAGE_SIZE,
    stack_size: int = DEFAULT_STACK_SIZE,
    render: bool = False,
    **overrides: Any,
) -> tuple[dict[str, Any], Circuit]:
    """Assemble a MetaDrive config for one circuit.

    Args:
        circuit: Circuit name or :class:`~drive_rl.circuits.Circuit`.
        obs_mode: ``"state"`` for vector observations, ``"camera"`` for a
            rendered RGB frame.
        image_size: Side length of the rendered camera observation.
        stack_size: Frames stacked into the camera observation.
        render: Open a 3D window. Training wants this off.
        **overrides: Any other MetaDrive config key, merged recursively.

    Returns:
        ``(config, circuit)``.

    Raises:
        ValueError: If ``obs_mode`` is not recognised.
    """
    if isinstance(circuit, str):
        circuit = get_circuit(circuit)
    if obs_mode not in ("state", "camera"):
        raise ValueError(f"obs_mode must be 'state' or 'camera', got {obs_mode!r}")

    config = deep_update(
        RACING_CONFIG, {"map_config": map_config(circuit), "use_render": render}
    )

    if obs_mode == "camera":
        from metadrive.component.sensors.rgb_camera import RGBCamera

        config["image_observation"] = True
        config["image_on_cuda"] = False
        config["stack_size"] = stack_size
        config["sensors"] = {"rgb_camera": (RGBCamera, image_size, image_size)}
        config["vehicle_config"] = deep_update(
            config["vehicle_config"], {"image_source": "rgb_camera"}
        )

    if overrides:
        config = deep_update(config, overrides)
    return config, circuit


def _make_env_class(circuit: Circuit):
    """Build a MetaDrive env subclass bound to one circuit."""
    from metadrive.envs.metadrive_env import MetaDriveEnv
    from metadrive.manager.pg_map_manager import PGMapManager

    map_class = build_map_class(circuit)

    class CircuitMapManager(PGMapManager):
        """Loads one circuit, once, and reuses it across episodes."""

        def reset(self) -> None:
            if len(self.spawned_objects) == 0:
                cfg = map_config(circuit)
                cfg.update(self.engine.global_config.get("map_config") or {})
                track = self.spawn_object(map_class, map_config=cfg, random_seed=None)
            else:
                track = next(iter(self.spawned_objects.values()))
            self.load_map(track)

    class RacingEnv(MetaDriveEnv):
        """Single-agent racing environment on a fixed circuit."""

        def setup_engine(self) -> None:
            super().setup_engine()
            self.engine.update_manager("map_manager", CircuitMapManager())

        @property
        def step_seconds(self) -> float:
            """Simulated seconds advanced by one environment step.

            MetaDrive runs ``decision_repeat`` physics substeps of
            ``physics_world_step_size`` seconds each, so one environment step is
            5 x 0.02 = 0.1 s. Reading this from the config rather than hardcoding it
            matters: getting it wrong scales every reported lap time.
            """
            return float(
            self.config.get("decision_repeat", 5)
            * self.config.get("physics_world_step_size", 0.02)
            )

        # -- lap tracking ------------------------------------------------- #

        def _reset_lap_state(self) -> None:
            self._previous_s: float | None = None
            self._laps_completed = 0
            self._offroad_steps = 0
            self._stuck_steps = 0
            self._simulated_time = 0.0
            self._lap_start_time = 0.0
            self.last_lap_time: float | None = None
            self.best_lap_time: float | None = None

        def _track_progress(self, position) -> tuple[float, float, bool]:
            """Project the vehicle, returning ``(delta, lateral, wrapped)``.

            A delta below minus half a lap means arc length wrapped from the end of
            the loop back to the start: the car crossed the start/finish line.
            """
            s, lateral = self.reference.project(position)
            previous = self._previous_s
            self._previous_s = s
            if previous is None:
                return 0.0, lateral, False

            delta = s - previous
            if delta < -0.5 * self.lap_length:
                return delta + self.lap_length, lateral, True
            if delta > 0.5 * self.lap_length:
                return delta - self.lap_length, lateral, False
            return delta, lateral, False

        # -- MetaDrive hooks ---------------------------------------------- #

        def reward_function(self, vehicle_id: str) -> tuple[float, dict[str, Any]]:
            """Racing reward: lap progress, penalised for leaving the line."""
            vehicle = self.agents[vehicle_id]
            self._simulated_time += self.step_seconds

            delta, lateral, wrapped = self._track_progress(vehicle.position)
            on_lane = bool(vehicle.on_lane)
            half_width = max(self.circuit.lane_width / 2.0, 1e-6)
            normalised_lateral = min(
                abs(lateral) / half_width, self.racing.lateral_penalty_cap
            )

            reward = self.racing.progress_reward * (delta / self.lap_length)
            reward -= self.racing.lateral_penalty * normalised_lateral**2

            if on_lane:
                self._offroad_steps = 0
            else:
                reward -= self.racing.offroad_penalty
                self._offroad_steps += 1

            speed_km_h = float(vehicle.speed_km_h)
            self._stuck_steps = (
                0 if speed_km_h > self.racing.stuck_speed_km_h else self._stuck_steps + 1
            )

            if wrapped:
                lap_time = self._simulated_time - self._lap_start_time
                self._laps_completed += 1
                self._lap_start_time = self._simulated_time
                self.last_lap_time = lap_time
                if self.best_lap_time is None or lap_time < self.best_lap_time:
                    self.best_lap_time = lap_time
                reward += self.racing.lap_bonus

            step_info = {
                "progress": delta,
                "lap_fraction": (self._previous_s or 0.0) / self.lap_length,
                "lateral": lateral,
                "normalised_lateral": normalised_lateral,
                "on_lane": on_lane,
                "lap_completed": wrapped,
                "laps_completed": self._laps_completed,
                "lap_time": self.last_lap_time,
                "best_lap_time": self.best_lap_time,
                "speed_km_h": speed_km_h,
            }
            return float(reward), step_info

        def done_function(self, vehicle_id: str) -> tuple[bool, dict[str, Any]]:
            """Terminate on sustained off-track driving, being stuck, or timeout."""
            if self._offroad_steps >= self.racing.offroad_tolerance:
                return True, {"off_road": True}
            if self._stuck_steps >= self.racing.stuck_tolerance:
                return True, {"stuck": True}
            if self._simulated_time >= self.racing.max_episode_seconds:
                return True, {"max_episode_time": True}
            return False, {}

        def cost_function(self, vehicle_id: str) -> tuple[float, dict[str, Any]]:
            """Off-track driving as a cost, for safe-RL use."""
            vehicle = self.agents[vehicle_id]
            off_road = not vehicle.on_lane
            return (0.1 if off_road else 0.0), {"off_road": bool(off_road)}

        def reset(self, seed: int | None = None):
            """Reset the environment and clear lap tracking."""
            self._reset_lap_state()
            return super().reset(seed=seed)

    RacingEnv.__name__ = f"RacingEnv_{circuit.name}"
    return RacingEnv


def make_racing_env(
    circuit: str | Circuit = DEFAULT_CIRCUIT,
    *,
    obs_mode: str = "state",
    image_size: int = DEFAULT_IMAGE_SIZE,
    stack_size: int = DEFAULT_STACK_SIZE,
    render: bool = False,
    racing: RacingParams | None = None,
    **overrides: Any,
):
    """Create a MetaDrive racing environment.

    Args:
        circuit: Circuit name or :class:`~drive_rl.circuits.Circuit`.
        obs_mode: ``"state"`` or ``"camera"``.
        image_size: Camera observation side length.
        stack_size: Frames stacked into the camera observation.
        render: Open a 3D window.
        racing: Reward and termination parameters.
        **overrides: Additional MetaDrive config keys, merged recursively.

    Returns:
        An environment exposing the standard MetaDrive/Gymnasium API, plus
        ``circuit``, ``reference``, ``lap_length`` and ``racing`` attributes.
    """
    config, resolved = build_racing_config(
        circuit,
        obs_mode=obs_mode,
        image_size=image_size,
        stack_size=stack_size,
        render=render,
        **overrides,
    )
    env = _make_env_class(resolved)(config)
    env.circuit = resolved
    env.reference = ReferenceLine(resolved)
    env.lap_length = env.reference.lap_length
    env.racing = racing or RacingParams()
    return env


class CameraObservation:
    """Adapt MetaDrive's stacked RGB observation for Stable-Baselines3.

    MetaDrive returns ``{"image": (H, W, 3, stack), "state": (n,)}``. A CNN
    policy wants a single channels-first array, so this flattens the frame to
    ``(stack * 3, H, W)`` and drops the state vector.
    """

    @staticmethod
    def to_channels_first(observation: dict[str, np.ndarray]) -> np.ndarray:
        """Return the stacked RGB frame as ``(stack * 3, H, W)`` float32.

        Args:
            observation: MetaDrive's observation dict.

        Returns:
            Channels-first float32 array.

        Raises:
            KeyError: If the observation has no ``image`` entry.
        """
        if "image" not in observation:
            raise KeyError(f"expected an 'image' key, got {sorted(observation)}")
        image = observation["image"]  # (H, W, C, stack)
        # (H, W, C, stack) -> (stack, H, W, C) -> (stack, C, H, W)
        stacked = np.transpose(image, (3, 2, 0, 1))
        height, width = stacked.shape[2], stacked.shape[3]
        return np.ascontiguousarray(stacked.reshape(-1, height, width), dtype=np.float32)
