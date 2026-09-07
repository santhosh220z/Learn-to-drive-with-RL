"""Deterministic evaluation harness for a trained PPO model.

Run:
    python -m drive_rl.evaluate --model models/ppo_carracing_<ts>/best_eval/best_model.zip
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from drive_rl.env import make_eval_env


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained model.")
    p.add_argument("--model", type=Path, required=True, help="Path to .zip model.")
    p.add_argument("--n-episodes", type=int, default=20, help="Eval episodes.")
    p.add_argument("--seed", type=int, default=42, help="Eval env seed.")
    p.add_argument("--out", type=Path, default=Path("eval.json"), help="Output JSON.")
    p.add_argument("--device", type=str, default="auto", help="Torch device.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.model.is_file():
        print(f"[eval] Model not found: {args.model}", file=sys.stderr)
        return 1

    from drive_rl.utils import select_device

    device = select_device(args.device)
    env = make_eval_env(seed=args.seed)
    model = PPO.load(str(args.model), env=env, device=device)

    rewards: list[float] = []
    lengths: list[int] = []
    lap_finished: list[bool] = []
    terminated: list[bool] = []

    obs = env.reset()
    episode_count = 0
    cur_reward = 0.0
    cur_len = 0
    while episode_count < args.n_episodes:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        cur_reward += float(reward[0])
        cur_len += 1
        if bool(done[0]):
            rewards.append(cur_reward)
            lengths.append(cur_len)
            ep_info = info[0].get("episode") or {}
            lap_finished.append(bool(ep_info.get("r", -1) and ep_info.get("r", 0) > 0))
            terminated.append(bool(info[0].get("terminal_observation") is not None))
            episode_count += 1
            cur_reward = 0.0
            cur_len = 0
            obs = env.reset()

    rewards_arr = np.asarray(rewards, dtype=np.float64)
    summary = {
        "model": str(args.model),
        "n_episodes": int(args.n_episodes),
        "mean_reward": float(rewards_arr.mean()),
        "std_reward": float(rewards_arr.std(ddof=1)) if len(rewards_arr) > 1 else 0.0,
        "min_reward": float(rewards_arr.min()),
        "max_reward": float(rewards_arr.max()),
        "mean_length": float(np.mean(lengths)),
        "episodes": [
            {
                "reward": float(r),
                "length": int(l),
            }
            for r, l in zip(rewards, lengths)
        ],
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(
        f"[eval] {args.n_episodes} episodes | "
        f"mean reward = {summary['mean_reward']:.2f} ± {summary['std_reward']:.2f} | "
        f"min={summary['min_reward']:.2f} max={summary['max_reward']:.2f} | "
        f"mean len = {summary['mean_length']:.0f}"
    )
    print(f"[eval] Wrote summary to {args.out}")
    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
