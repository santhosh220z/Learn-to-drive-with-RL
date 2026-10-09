"""Watch a trained policy drive the circuit in a 3D window.

    # follow the newest best checkpoint of a run while it trains
    python -m drive_rl.watch --run-dir models/<run>

    # watch one specific model
    python -m drive_rl.watch --model models/<run>/best_eval/best_model.zip --loop

Run this in a second terminal alongside ``python -m drive_rl.train``. It polls
the run directory and hot-reloads whenever a better checkpoint is saved, so you
can watch the policy improve without stopping training.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from drive_rl.utils import select_device


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Watch a racing policy drive in 3D.")
    p.add_argument("--run-dir", type=Path, default=None, help="Run directory to poll.")
    p.add_argument("--model", type=Path, default=None, help="A specific model to roll out.")
    p.add_argument("--circuit", default=None, help="Circuit (default: from config snapshot).")
    p.add_argument("--obs-mode", default=None, choices=["state", "camera"])
    p.add_argument("--image-size", type=int, default=84)
    p.add_argument("--stack-size", type=int, default=3)
    p.add_argument("--poll-interval", type=float, default=10.0, help="Seconds between checks.")
    p.add_argument("--max-steps", type=int, default=6000, help="Step cap per episode.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--window", type=int, nargs=2, default=(1100, 750), help="Window size.")
    p.add_argument("--loop", action="store_true", help="Replay forever (with --model).")
    args = p.parse_args(argv)

    if (args.run_dir is None) == (args.model is None):
        p.error("provide exactly one of --run-dir or --model")
    return args


def best_candidate(run_dir: Path) -> Path | None:
    """Best available checkpoint in a run, preferring evaluation-best."""
    for candidate in (
        run_dir / "best_eval" / "best_model.zip",
        run_dir / "best_model_by_train_reward.zip",
        run_dir / "final_model.zip",
    ):
        if candidate.is_file():
            return candidate
    return None


def load_settings(model_path: Path, args: argparse.Namespace) -> tuple[str, str]:
    """Recover circuit and observation mode from the run's config snapshot."""
    snapshot = model_path.parent / "config_snapshot.json"
    if not snapshot.is_file():
        return args.circuit or "suzuka", args.obs_mode or "state"
    try:
        data = json.loads(snapshot.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return args.circuit or "suzuka", args.obs_mode or "state"
    return (
        args.circuit or data.get("circuit", "suzuka"),
        args.obs_mode or data.get("obs_mode", "state"),
    )


def rollout(args: argparse.Namespace, model_path: Path) -> float:
    """Render one episode in a 3D window; returns the undiscounted return."""
    from stable_baselines3 import PPO

    from drive_rl.racing_env import make_racing_vec_env

    circuit, obs_mode = load_settings(model_path, args)
    env = make_racing_vec_env(
        1,
        circuit=circuit,
        obs_mode=obs_mode,
        seed=args.seed,
        image_size=args.image_size,
        stack_size=args.stack_size,
        vec_cls="dummy",
        norm_obs=obs_mode == "state",
        norm_reward=False,
        render=True,
    )
    try:
        model = PPO.load(str(model_path), env=env, device=select_device(args.device))
    except ValueError as exc:
        env.close()
        print(f"[watch] {model_path.name} does not match this env ({exc}).", file=sys.stderr)
        return float("nan")

    observation = env.reset()
    total = 0.0
    laps = 0
    try:
        for _ in range(args.max_steps):
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, done, infos = env.step(action)
            total += float(reward[0])
            info = infos[0] if isinstance(infos, list) else infos
            if info.get("lap_completed"):
                laps += 1
                if info.get("lap_time"):
                    print(f"[watch]   lap {laps}: {float(info['lap_time']):.2f} s")
            env.render()
            if bool(done[0]):
                break
    finally:
        env.close()
    return total


def watch_run(args: argparse.Namespace) -> int:
    if not args.run_dir.is_dir():
        print(f"[watch] run dir not found: {args.run_dir}", file=sys.stderr)
        return 1
    print(f"[watch] polling {args.run_dir} every {args.poll_interval:.0f}s. Ctrl+C to stop.")

    current: Path | None = None
    stamp = -1.0
    while True:
        candidate = best_candidate(args.run_dir)
        if candidate is None:
            if current is None:
                print("[watch] no checkpoint yet; waiting.")
            time.sleep(args.poll_interval)
            continue
        mtime = candidate.stat().st_mtime
        if candidate != current or mtime != stamp:
            current, stamp = candidate, mtime
            print(f"[watch] loading {candidate.name} ({candidate.stat().st_size // 1024} KiB)")
            total = rollout(args, candidate)
            if total == total:  # not NaN
                print(f"[watch] episode return {total:.2f} ({current.name})")
        time.sleep(args.poll_interval)


def rollout_once(args: argparse.Namespace) -> int:
    if not args.model.is_file():
        print(f"[watch] model not found: {args.model}", file=sys.stderr)
        return 1
    while True:
        total = rollout(args, args.model)
        if total == total:
            print(f"[watch] episode return {total:.2f}")
        if not args.loop:
            break
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.run_dir is not None:
            return watch_run(args)
        return rollout_once(args)
    except KeyboardInterrupt:
        print("\n[watch] stopped.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
