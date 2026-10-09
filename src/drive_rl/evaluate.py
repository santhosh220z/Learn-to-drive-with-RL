"""Evaluate a trained racing policy and report lap times.

    python -m drive_rl.evaluate --model models/<run>/best_eval/best_model.zip
    python -m drive_rl.evaluate --model ... --circuit silverstone --episodes 20

Lap time is the headline number, but it is not the whole story on a fixed track:
a policy can turn a quick lap by cutting across the run-off. This reports the
line deviation and off-track fraction alongside the time so a fast lap that left
the road is visible rather than celebrated.

Writes ``eval.json`` next to the run and prints a summary.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

from drive_rl.utils import select_device, set_global_seed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a racing policy on a circuit.")
    p.add_argument("--model", type=Path, required=True, help="Model file to evaluate.")
    p.add_argument("--circuit", default=None, help="Circuit (defaults to the training circuit).")
    p.add_argument("--episodes", type=int, default=10, help="Number of episodes.")
    p.add_argument("--seed", type=int, default=0, help="Evaluation seed.")
    p.add_argument("--device", default="auto", help="Torch device.")
    p.add_argument(
        "--obs-mode",
        default=None,
        choices=["state", "camera"],
        help="Observation mode (defaults to the training circuit's).",
    )
    p.add_argument("--image-size", type=int, default=84)
    p.add_argument("--stack-size", type=int, default=3)
    p.add_argument("--out", type=Path, default=None, help="Where to write eval.json.")
    p.add_argument("--max-steps", type=int, default=20_000, help="Step cap per episode.")
    p.add_argument(
        "--require-laps",
        type=int,
        default=0,
        help="Minimum completed laps for an episode to count toward lap-time stats.",
    )
    return p.parse_args(argv)


def infer_circuit(model_path: Path, override: str | None) -> str:
    """Recover the training circuit from the run's config snapshot if present."""
    if override:
        return override
    snapshot = model_path.parent / "config_snapshot.json"
    if not snapshot.is_file():
        return "suzuka"
    try:
        return json.loads(snapshot.read_text(encoding="utf-8")).get("circuit", "suzuka")
    except (json.JSONDecodeError, OSError):
        return "suzuka"


def infer_obs_mode(snapshot: Path, override: str | None) -> str:
    if override:
        return override
    if not snapshot.is_file():
        return "state"
    try:
        return json.loads(snapshot.read_text(encoding="utf-8")).get("obs_mode", "state")
    except (json.JSONDecodeError, OSError):
        return "state"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.model.is_file():
        print(f"[evaluate] model not found: {args.model}", file=sys.stderr)
        return 1

    set_global_seed(args.seed)

    from stable_baselines3 import PPO

    from drive_rl.racing_env import make_racing_vec_env

    snapshot = args.model.parent / "config_snapshot.json"
    circuit = infer_circuit(args.model, args.circuit)
    obs_mode = infer_obs_mode(snapshot, args.obs_mode)

    env = make_racing_vec_env(
        1,
        circuit=circuit,
        obs_mode=obs_mode,
        seed=args.seed,
        image_size=args.image_size,
        stack_size=args.stack_size,
        vec_cls="dummy",
    )
    model = PPO.load(str(args.model), env=env, device=select_device(args.device))

    lap_times: list[float] = []
    returns: list[float] = []
    off_track: list[float] = []
    max_lateral: list[float] = []
    laps_per_episode: list[int] = []

    for episode in range(args.episodes):
        observation = env.reset()
        total_reward = 0.0
        steps = 0
        episode_off = 0
        episode_max_lateral = 0.0
        episode_laps = 0
        started = time.perf_counter()

        for _ in range(args.max_steps):
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, done, infos = env.step(action)
            total_reward += float(reward[0])
            steps += 1
            info = infos[0] if isinstance(infos, list) else infos
            if info.get("on_lane") is False:
                episode_off += 1
            episode_max_lateral = max(
                episode_max_lateral, abs(float(info.get("lateral", 0.0)))
            )
            if info.get("lap_completed"):
                episode_laps += 1
                if info.get("lap_time"):
                    lap_times.append(float(info["lap_time"]))
            if bool(done[0]):
                break

        returns.append(total_reward)
        off_track.append(episode_off / max(steps, 1))
        max_lateral.append(episode_max_lateral)
        laps_per_episode.append(episode_laps)
        print(
            f"[evaluate] episode {episode + 1}/{args.episodes}: "
            f"laps={episode_laps} reward={total_reward:.2f} "
            f"off-track={100 * episode_off / max(steps, 1):.1f}% "
            f"max line deviation={episode_max_lateral:.2f} m "
            f"({time.perf_counter() - started:.1f}s)"
        )

    env.close()

    summary = {
        "model": str(args.model),
        "circuit": circuit,
        "obs_mode": obs_mode,
        "episodes": args.episodes,
        "seed": args.seed,
        "laps_completed": len(lap_times),
        "mean_laps_per_episode": statistics.fmean(laps_per_episode),
        "mean_return": statistics.fmean(returns),
        "stdev_return": statistics.pstdev(returns) if len(returns) > 1 else 0.0,
        "lap_time": {
            "mean": statistics.fmean(lap_times) if lap_times else None,
            "best": min(lap_times) if lap_times else None,
            "worst": max(lap_times) if lap_times else None,
            "count": len(lap_times),
        },
        "off_track_fraction": {
            "mean": statistics.fmean(off_track),
            "worst": max(off_track),
        },
        "max_lateral_m": {
            "mean": statistics.fmean(max_lateral),
            "worst": max(max_lateral),
        },
    }

    out_path = args.out or (args.model.parent / "eval.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lap = summary["lap_time"]
    print("")
    print(f"[evaluate] circuit      : {circuit}")
    print(f"[evaluate] episodes     : {args.episodes}")
    print(f"[evaluate] laps/episode : {summary['mean_laps_per_episode']:.2f}")
    if lap["mean"] is not None:
        print(
            f"[evaluate] lap time     : {lap['mean']:.2f} s mean "
            f"(best {lap['best']:.2f}, worst {lap['worst']:.2f}, n={lap['count']})"
        )
    else:
        print("[evaluate] lap time     : no laps completed")
    print(f"[evaluate] mean return  : {summary['mean_return']:.2f}")
    print(f"[evaluate] off-track    : {100 * summary['off_track_fraction']['mean']:.1f}% mean")
    print(f"[evaluate] line dev.    : {summary['max_lateral_m']['worst']:.2f} m worst")
    print(f"[evaluate] written to   : {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
