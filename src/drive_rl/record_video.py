"""Record an MP4 of one trained-policy rollout for the README demo.

Run:
    python -m drive_rl.record_video --model models/.../best_eval/best_model.zip
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from stable_baselines3 import PPO

from drive_rl.env import ENV_ID, make_eval_env
from drive_rl.utils import select_device


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Record a video of a trained agent.")
    p.add_argument("--model", type=Path, required=True, help="Path to .zip model.")
    p.add_argument("--out-dir", type=Path, default=Path("videos"), help="Output dir.")
    p.add_argument("--name", type=str, default="episode", help="Video base name.")
    p.add_argument("--max-steps", type=int, default=1500, help="Step cap.")
    p.add_argument("--seed", type=int, default=0, help="Eval seed.")
    p.add_argument("--device", type=str, default="auto", help="Torch device.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.model.is_file():
        print(f"[record] Model not found: {args.model}", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Build a NON-stacked, RGB-array env so the recorder sees clean frames.
    import gymnasium as gym
    from gymnasium.wrappers import RecordVideo

    env = gym.make(ENV_ID, continuous=True, render_mode="rgb_array")
    env = RecordVideo(
        env,
        video_folder=str(args.out_dir),
        name_prefix=args.name,
        episode_trigger=lambda ep: True,
    )

    device = select_device(args.device)
    model = PPO.load(str(args.model), device=device)

    obs, _ = env.reset(seed=args.seed)
    total_reward = 0.0
    for step in range(args.max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)
        if terminated or truncated:
            break
    env.close()
    print(f"[record] Recorded {step + 1} steps | total reward = {total_reward:.2f}")
    print(f"[record] Output dir: {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
