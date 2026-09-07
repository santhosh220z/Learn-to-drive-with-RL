"""Optional Gymnasium wrappers for shaping CarRacing.

Both wrappers are **opt-in** and off by default in the training script. They
exist to enable ablations against the native reward, not to silently alter it.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
from gymnasium import Env, Wrapper


class ActionSmoothingWrapper(Wrapper):
    """Exponentially smoothed actions: `a_t = (1-α)·a_{t-1} + α·a_raw`.

    CarRacing agents tend to jitter the steering between frames. Smoothing
    reduces oscillation and can make learned policies more stable.

    Args:
        env: Source environment.
        alpha: Smoothing factor in `(0, 1]`. `1.0` is a passthrough; `0.1`
            heavily smooths. Defaults to `0.5`.
    """

    def __init__(self, env: Env, alpha: float = 0.5) -> None:
        super().__init__(env)
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self.alpha = float(alpha)
        self._prev_action: np.ndarray | None = None

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        self._prev_action = np.zeros(self.action_space.shape, dtype=np.float32)
        return obs, info

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        if self._prev_action is None:
            self._prev_action = np.zeros_like(action)
        smoothed = (1.0 - self.alpha) * self._prev_action + self.alpha * action
        # Clip into the env's action bounds to stay safe.
        smoothed = np.clip(smoothed, self.action_space.low, self.action_space.high)
        self._prev_action = smoothed
        return self.env.step(smoothed)


class RewardShapingWrapper(Wrapper):
    """Lightweight reward shaping. **Off by default**; the native reward is
    the honest signal. This wrapper exists to explore two hypotheses:

    1. Penalize the magnitude of steering change (reduces oscillation).
    2. Reward forward velocity proxy (greener pixels → more progress).

    Args:
        env: Source environment. Expected to be a single CarRacing env with
            an `unwrapped.car` attribute carrying `linearVelocity` (Box2D).
        steer_penalty: Coefficient for `|Δsteer|` per step.
        forward_bonus: Coefficient for forward speed reward per step.
    """

    def __init__(
        self,
        env: Env,
        steer_penalty: float = 0.0,
        forward_bonus: float = 0.0,
    ) -> None:
        super().__init__(env)
        self.steer_penalty = float(steer_penalty)
        self.forward_bonus = float(forward_bonus)
        self._prev_steer = 0.0

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        self._prev_steer = 0.0
        return obs, info

    def step(self, action: np.ndarray):
        obs, reward, terminated, truncated, info = self.env.step(action)
        shaped = float(reward)

        if self.steer_penalty > 0.0 and action.ndim > 0:
            steer = float(action[0])
            shaped -= self.steer_penalty * abs(steer - self._prev_steer)
            self._prev_steer = steer

        if self.forward_bonus > 0.0:
            speed = self._car_speed()
            if speed is not None:
                shaped += self.forward_bonus * speed

        return obs, shaped, terminated, truncated, info

    def _car_speed(self) -> float | None:
        """Return scalar forward speed if a Box2D car is reachable."""
        try:
            car = self.unwrapped.car
            vx, vy = car.linearVelocity
            # Project onto the car's heading so backward motion doesn't reward.
            import math

            fx, fy = car.hull.GetWorldVector((1.0, 0.0))
            return float(vx * fx + vy * fy) * math.copysign(1.0, fx)
        except AttributeError:
            return None


class FrameSkipWrapper(Wrapper):
    """Repeat an action for N frames and sum rewards.

    CarRacing already steps every frame; this is exposed for ablations. Not
    used by the default training pipeline.
    """

    def __init__(self, env: Env, skip: int = 4) -> None:
        super().__init__(env)
        if skip < 1:
            raise ValueError(f"skip must be >= 1, got {skip}")
        self.skip = skip

    def step(self, action: np.ndarray):
        total_reward = 0.0
        terminated = truncated = False
        info: dict[str, Any] = {}
        obs = None
        for _ in range(self.skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info
