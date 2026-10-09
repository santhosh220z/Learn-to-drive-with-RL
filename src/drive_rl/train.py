"""Train a racing policy with PPO on MetaDrive.

    # quickest way to see the pipeline work
    python -m drive_rl.train --preset smoke

    # the usual first real run
    python -m drive_rl.train --preset balanced

    # check shapes and throughput without training anything
    python -m drive_rl.train --preset balanced --dry-run

Training is deliberately not started implicitly. Every run prints a plan,
measures the environment's real throughput, converts that into an estimated
wall-clock time, and asks for confirmation first -- a "2.5 hour" run that was
really going to take two days is worth catching before it starts rather than
after.

Throughput varies enormously between the two observation modes (a rendered 3D
frame is far more expensive than a lidar vector), and varies again with how many
environments the machine can actually keep alive. Rather than guess, the estimate
below is measured on the environment that was just built.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from drive_rl.utils import save_config_snapshot, select_device, set_global_seed

# Presets differ in how much the user is willing to spend, not in what the task
# is. `timesteps` for `balanced` and `quality` are deliberately round: they are
# launch points for further training, not claims about convergence.
PRESETS: dict[str, dict[str, Any]] = {
    "smoke": {
        "timesteps": 10_000,
        "n_envs": 2,
        "n_steps": 256,
        "batch_size": 64,
        "n_epochs": 4,
        "learning_rate": 3e-4,
        "obs_mode": "state",
        "eval_freq": 5_000,
        "note": "Pipeline check only. The policy will not drive.",
    },
    "fast": {
        "timesteps": 300_000,
        "n_envs": 4,
        "n_steps": 1024,
        "batch_size": 128,
        "n_epochs": 10,
        "learning_rate": 3e-4,
        "obs_mode": "state",
        "eval_freq": 25_000,
        "note": "Cheap state-observation run; usually enough to complete laps slowly.",
    },
    "balanced": {
        "timesteps": 1_500_000,
        "n_envs": 4,
        "n_steps": 2048,
        "batch_size": 128,
        "n_epochs": 10,
        "learning_rate": 2.5e-4,
        "obs_mode": "state",
        "eval_freq": 50_000,
        "note": "Default. Strong state-observation baseline on one circuit.",
    },
    "camera": {
        "timesteps": 1_000_000,
        "n_envs": 4,
        "n_steps": 1024,
        "batch_size": 128,
        "n_epochs": 10,
        "learning_rate": 2.5e-4,
        "obs_mode": "camera",
        "eval_freq": 50_000,
        "note": "3D camera observations. Much slower per step than state.",
    },
    "quality": {
        "timesteps": 4_000_000,
        "n_envs": 4,
        "n_steps": 2048,
        "batch_size": 128,
        "n_epochs": 10,
        "learning_rate": 2.5e-4,
        "obs_mode": "state",
        "eval_freq": 100_000,
        "note": "Long run. Expect this to take many hours; resume from a checkpoint.",
    },
}

HP_KEYS = ("n_steps", "batch_size", "n_epochs", "learning_rate")


@dataclass
class Settings:
    """Fully resolved training settings."""

    preset: str
    circuit: str
    obs_mode: str
    timesteps: int
    n_envs: int
    seed: int
    device: str
    eval_freq: int
    n_eval_episodes: int
    hyperparams: dict[str, Any] = field(default_factory=dict)
    image_size: int = 84
    stack_size: int = 3
    save_dir: Path = Path("models")
    run_name: str | None = None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a MetaDrive racing policy with PPO.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--preset", default="balanced", choices=sorted(PRESETS), help="Preset.")
    p.add_argument("--circuit", default="suzuka", help="Circuit name.")
    p.add_argument(
        "--obs-mode",
        default=None,
        choices=["state", "camera"],
        help="Override the preset's observation mode.",
    )
    p.add_argument("--timesteps", type=int, default=None, help="Override total timesteps.")
    p.add_argument("--n-envs", type=int, default=None, help="Override parallel environments.")
    p.add_argument("--seed", type=int, default=0, help="Base seed.")
    p.add_argument("--device", default="auto", help="Torch device.")
    p.add_argument("--eval-freq", type=int, default=None, help="Evaluate every N steps.")
    p.add_argument("--n-eval-episodes", type=int, default=3, help="Episodes per evaluation.")
    p.add_argument("--save-dir", type=Path, default=Path("models"), help="Output directory.")
    p.add_argument("--run-name", default=None, help="Run subdirectory (timestamped if unset).")
    p.add_argument("--image-size", type=int, default=84, help="Camera observation size.")
    p.add_argument("--stack-size", type=int, default=3, help="Stacked camera frames.")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Build environments, print shapes and measured throughput, then exit.",
    )
    p.add_argument("--yes", action="store_true", help="Skip the confirmation prompt.")
    for key in HP_KEYS:
        p.add_argument(f"--{key.replace('_', '-')}", type=float, default=None, help=f"Override {key}.")
    return p.parse_args(argv)


def resolve_settings(args: argparse.Namespace) -> Settings:
    """Merge preset defaults with command-line overrides."""
    preset = PRESETS[args.preset]
    settings = Settings(
        preset=args.preset,
        circuit=args.circuit,
        obs_mode=args.obs_mode or preset["obs_mode"],
        timesteps=args.timesteps or preset["timesteps"],
        n_envs=args.n_envs or preset["n_envs"],
        seed=args.seed,
        device=args.device,
        eval_freq=args.eval_freq or preset["eval_freq"],
        n_eval_episodes=args.n_eval_episodes,
        image_size=args.image_size,
        stack_size=args.stack_size,
        save_dir=args.save_dir,
        run_name=args.run_name,
    )
    hyperparams = {
        "n_steps": int(preset["n_steps"]),
        "batch_size": int(preset["batch_size"]),
        "n_epochs": int(preset["n_epochs"]),
        "learning_rate": float(preset["learning_rate"]),
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.0,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
    }
    for key in HP_KEYS:
        override = getattr(args, key)
        if override is not None:
            hyperparams[key] = int(override) if key != "learning_rate" else float(override)
    settings.hyperparams = hyperparams
    return settings


def build_run_dir(settings: Settings) -> Path:
    settings.save_dir.mkdir(parents=True, exist_ok=True)
    name = settings.run_name or f"ppo_{settings.circuit}_{settings.obs_mode}_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir = settings.save_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def policy_for(settings: Settings) -> str:
    """CnnPolicy for camera observations, MlpPolicy for state observations."""
    return "CnnPolicy" if settings.obs_mode == "camera" else "MlpPolicy"


def measure_throughput(env, steps: int = 60) -> float:
    """Step a zero-action environment and return steps per second.

    Measured on the real environment rather than assumed, because throughput
    varies by an order of magnitude between state and camera observations.
    """
    import numpy as np

    n_envs = env.num_envs
    action = np.zeros((n_envs, env.action_space.shape[0]), dtype=np.float32)
    env.reset()
    for _ in range(5):
        env.step(action)
    start = time.perf_counter()
    for _ in range(steps):
        env.step(action)
    elapsed = time.perf_counter() - start
    return (steps * n_envs) / max(elapsed, 1e-9)


def format_duration(hours: float) -> str:
    if hours < 1:
        return f"{hours * 60:.0f} min"
    if hours < 48:
        return f"{hours:.1f} h"
    return f"{hours / 24:.1f} days"


def describe_plan(settings: Settings, env, steps_per_second: float) -> str:
    hours = settings.timesteps / max(steps_per_second, 1e-9) / 3600
    return "\n".join(
        [
            "",
            "  Run plan",
            f"    preset        : {settings.preset} ({PRESETS[settings.preset]['note']})",
            f"    circuit       : {settings.circuit}",
            f"    observation   : {settings.obs_mode}  ->  {policy_for(settings)}",
            f"    envs          : {settings.n_envs}",
            f"    timesteps     : {settings.timesteps:,}",
            f"    obs space     : {env.observation_space}",
            f"    action space  : {env.action_space}",
            f"    measured rate : {steps_per_second:.0f} steps/s",
            f"    estimated time: ~{format_duration(hours)} (environment only)",
            "",
        ]
    )


def confirm(prompt: str) -> bool:
    try:
        answer = input(prompt).strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False
    return answer in ("y", "yes")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = resolve_settings(args)
    set_global_seed(settings.seed)

    from drive_rl.callbacks import make_callback_list
    from drive_rl.racing_env import make_racing_vec_env

    run_dir = build_run_dir(settings)
    device = select_device(settings.device)
    print(f"[train] run dir: {run_dir}")

    train_env = make_racing_vec_env(
        settings.n_envs,
        circuit=settings.circuit,
        obs_mode=settings.obs_mode,
        seed=settings.seed,
        image_size=settings.image_size,
        stack_size=settings.stack_size,
    )
    eval_env = make_racing_vec_env(
        1,
        circuit=settings.circuit,
        obs_mode=settings.obs_mode,
        seed=settings.seed + 10_000,
        image_size=settings.image_size,
        stack_size=settings.stack_size,
        vec_cls="dummy",
    )

    steps_per_second = measure_throughput(train_env)
    print(describe_plan(settings, train_env, steps_per_second))

    save_config_snapshot(
        {
            "preset": settings.preset,
            "circuit": settings.circuit,
            "obs_mode": settings.obs_mode,
            "timesteps": settings.timesteps,
            "n_envs": settings.n_envs,
            "seed": settings.seed,
            "device": device,
            "eval_freq": settings.eval_freq,
            "image_size": settings.image_size,
            "stack_size": settings.stack_size,
            "hyperparams": settings.hyperparams,
            "measured_steps_per_second": round(steps_per_second, 1),
            "run_dir": str(run_dir),
        },
        run_dir / "config_snapshot.json",
    )

    if args.dry_run:
        print("[train] --dry-run: not training. Done.")
        train_env.close()
        eval_env.close()
        return 0

    hours = settings.timesteps / max(steps_per_second, 1e-9) / 3600
    if hours > 0.5 and not args.yes:
        print("[train] This is a long run. It will not start without confirmation.")
        if not confirm(f"[train] Start a ~{format_duration(hours)} run? [y/N] "):
            print("[train] Cancelled.")
            train_env.close()
            eval_env.close()
            return 0

    from stable_baselines3 import PPO

    model = PPO(
        policy_for(settings),
        train_env,
        verbose=1,
        seed=settings.seed,
        device=device,
        tensorboard_log=str(run_dir / "tb"),
        **settings.hyperparams,
    )

    callbacks = make_callback_list(
        eval_env=eval_env,
        eval_freq=max(settings.eval_freq // settings.n_envs, 1),
        save_path=run_dir,
        best_model_path=run_dir / "best_eval",
        n_eval_episodes=settings.n_eval_episodes,
        verbose=1,
    )

    print(f"[train] Training {settings.timesteps:,} steps on {device}...")
    try:
        model.learn(
            total_timesteps=settings.timesteps,
            callback=callbacks,
            progress_bar=False,
            tb_log_name="PPO",
        )
    except KeyboardInterrupt:
        print("\n[train] Interrupted; saving partial model.")
    finally:
        model.save(run_dir / "final_model.zip")
        train_env.close()
        eval_env.close()
        print(f"[train] saved {run_dir / 'final_model.zip'}")
        print(f"[train] tensorboard --logdir {run_dir / 'tb'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
