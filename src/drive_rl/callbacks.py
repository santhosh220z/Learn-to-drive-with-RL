"""Custom SB3 callbacks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from stable_baselines3.common.callbacks import BaseCallback


class SaveBestModelByReward(BaseCallback):
    """Save the best model by mean training episode reward.

    Useful when paired with `EvalCallback` for raw evaluation rewards: this
    callback watches `rollout/ep_rew_mean` from the SB3 logger and copies the
    model to `save_path/best_model.zip` whenever the mean improves.
    """

    def __init__(self, save_path: Path, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.best_mean_reward = -float("inf")

    def _on_step(self) -> bool:
        # Read the latest ep_rew_mean from the logger, if available.
        ep_rew_mean = self.model.logger.name_to_value.get("rollout/ep_rew_mean")
        if ep_rew_mean is None:
            return True
        if ep_rew_mean > self.best_mean_reward:
            self.best_mean_reward = float(ep_rew_mean)
            best_path = self.save_path / "best_model_by_train_reward.zip"
            self.model.save(best_path)
            if self.verbose:
                print(f"[SaveBest] new best train reward {ep_rew_mean:.2f} -> {best_path}")
        return True


def make_callback_list(
    eval_env: Any,
    eval_freq: int,
    save_path: Path,
    best_model_path: Path,
    n_eval_episodes: int = 5,
    verbose: int = 1,
) -> Any:
    """Compose EvalCallback + SaveBestModelByReward into a single list."""
    from stable_baselines3.common.callbacks import EvalCallback

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(best_model_path),
        log_path=str(save_path / "eval_logs"),
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
        verbose=verbose,
    )
    best_cb = SaveBestModelByReward(save_path=save_path, verbose=verbose)
    return [eval_cb, best_cb]
