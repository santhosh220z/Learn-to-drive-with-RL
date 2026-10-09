"""Record a 3D video of a trained policy driving the circuit.

    python -m drive_rl.record_video --model models/<run>/best_eval/best_model.zip

Renders the environment offscreen and writes an MP4. Rendering costs more than
the policy does, so this is for producing demos, not for training.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from drive_rl.utils import select_device


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Record a 3D video of a racing policy.")
    p.add_argument("--model", type=Path, required=True, help="Model file to roll out.")
    p.add_argument("--out-dir", type=Path, default=Path("videos"), help="Output directory.")
    p.add_argument("--name", default="lap", help="Output filename stem.")
    p.add_argument("--circuit", default=None, help="Circuit (default: from config snapshot).")
    p.add_argument("--obs-mode", default=None, choices=["state", "camera"])
    p.add_argument("--image-size", type=int, default=84)
    p.add_argument("--stack-size", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=3000, help="Step cap.")
    p.add_argument("--fps", type=int, default=30, help="Output frame rate.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    return p.parse_args(argv)


def load_settings(model_path: Path, args: argparse.Namespace) -> tuple[str, str]:
    import json

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


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.model.is_file():
        print(f"[record] model not found: {args.model}", file=sys.stderr)
        return 1

    from stable_baselines3 import PPO

    from drive_rl.racing_env import make_racing_vec_env

    circuit, obs_mode = load_settings(args.model, args)

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
    )
    model = PPO.load(str(args.model), env=env, device=select_device(args.device))

    try:
        from stable_baselines3.common.vec_env import VecVideoRecorder

        args.out_dir.mkdir(parents=True, exist_ok=True)
        # VecVideoRecorder captures on reset, so wrapping before the first reset
        # records from the start of the episode.
        video_env = VecVideoRecorder(
            env,
            video_folder=str(args.out_dir),
            record_video_trigger=lambda _episode: True,
            video_length=args.max_steps,
            name_prefix=args.name,
        )
        observation = video_env.reset()
        laps = 0
        recorded = 0
        for _step in range(args.max_steps):
            action, _ = model.predict(observation, deterministic=True)
            observation, _reward, done, infos = video_env.step(action)
            recorded += 1
            info = infos[0] if isinstance(infos, list) else infos
            if info.get("lap_completed"):
                laps += 1
                print(f"[record] lap {laps}: {float(info['lap_time']):.2f} s")
            if bool(done[0]):
                break
        print(f"[record] recorded {recorded} steps ({laps} laps) to {args.out_dir}")
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
