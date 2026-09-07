"""Training entrypoint: PPO on CarRacing-v3.

Run a quick smoke test:
    python -m drive_rl.train --timesteps 10000 --n-envs 2

Run a serious baseline:
    python -m drive_rl.train --timesteps 1000000 --n-envs 8 --seed 0
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure

from drive_rl.callbacks import make_callback_list
from drive_rl.env import make_env, make_eval_env
from drive_rl.utils import save_config_snapshot, select_device, set_global_seed

DEFAULT_HYPERPARAMS: dict[str, Any] = {
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.0,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train PPO on CarRacing-v3.")
    p.add_argument("--algo", default="ppo", choices=["ppo"], help="Algorithm.")
    p.add_argument("--timesteps", type=int, default=1_000_000, help="Total env steps.")
    p.add_argument("--n-envs", type=int, default=8, help="Parallel training envs.")
    p.add_argument("--seed", type=int, default=0, help="Base seed.")
    p.add_argument("--save-dir", type=Path, default=Path("models"), help="Output dir.")
    p.add_argument("--run-name", type=str, default=None, help="Subdir name (auto if None).")
    p.add_argument("--eval-freq", type=int, default=10_000, help="Eval every N steps.")
    p.add_argument("--n-eval-episodes", type=int, default=5, help="Episodes per eval.")
    p.add_argument("--device", type=str, default="auto", help="Torch device.")
    p.add_argument(
        "--vec-cls",
        type=str,
        default="auto",
        choices=["auto", "dummy", "subproc"],
        help="VecEnv class. 'subproc' recommended for n_envs > 1.",
    )
    # Hyperparameters (override defaults)
    for k, v in DEFAULT_HYPERPARAMS.items():
        p.add_argument(f"--{k.replace('_', '-')}", type=type(v), default=v)
    return p.parse_args(argv)


def build_run_dir(save_dir: Path, run_name: str | None) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    name = run_name or f"ppo_carracing_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = save_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    set_global_seed(args.seed)
    device = select_device(args.device)

    run_dir = build_run_dir(args.save_dir, args.run_name)
    tb_log = run_dir / "tb"
    tb_log.mkdir(parents=True, exist_ok=True)

    # Build envs
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    vec_cls = {
        "auto": SubprocVecEnv if args.n_envs > 1 else DummyVecEnv,
        "dummy": DummyVecEnv,
        "subproc": SubprocVecEnv,
    }[args.vec_cls]

    print(f"[train] Building {args.n_envs} CarRacing-v3 envs ({vec_cls.__name__})...")
    train_env = make_env(
        n_envs=args.n_envs,
        seed=args.seed,
        vec_cls=vec_cls,
    )
    eval_env = make_eval_env(seed=args.seed + 1000)

    # Compose hyperparams from defaults + overrides
    hp = {k: getattr(args, k) for k in DEFAULT_HYPERPARAMS}
    print(f"[train] Hyperparameters: {hp}")

    model = PPO(
        "CnnPolicy",
        train_env,
        verbose=1,
        seed=args.seed,
        device=device,
        tensorboard_log=str(tb_log),
        **hp,
    )
    model._logger = configure(str(run_dir / "sb3_logs"), ["stdout", "tensorboard"])

    callbacks = make_callback_list(
        eval_env=eval_env,
        eval_freq=max(args.eval_freq // args.n_envs, 1),
        save_path=run_dir,
        best_model_path=run_dir / "best_eval",
        n_eval_episodes=args.n_eval_episodes,
        verbose=1,
    )

    save_config_snapshot(
        {
            "algo": args.algo,
            "timesteps": args.timesteps,
            "n_envs": args.n_envs,
            "seed": args.seed,
            "device": device,
            "vec_cls": args.vec_cls,
            "eval_freq": args.eval_freq,
            "n_eval_episodes": args.n_eval_episodes,
            "hyperparameters": hp,
            "run_dir": str(run_dir),
        },
        run_dir / "config_snapshot.json",
    )

    print(f"[train] Training for {args.timesteps:,} timesteps...")
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callbacks,
            progress_bar=False,
            tb_log_name="PPO",
        )
    except KeyboardInterrupt:
        print("\n[train] Interrupted. Saving partial model...")
    finally:
        final_path = run_dir / "final_model.zip"
        model.save(final_path)
        train_env.close()
        eval_env.close()
        print(f"[train] Saved final model to {final_path}")
        print(f"[train] TensorBoard: tensorboard --logdir {tb_log}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
