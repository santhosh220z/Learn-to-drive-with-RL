# Learn to Drive with RL

Teach a reinforcement-learning agent to drive a racing line, on **MetaDrive** with **PPO**
from [Stable-Baselines3](https://stable-baselines3.readthedocs.io/).

The agent laps real circuits — Suzuka, Silverstone, Monaco — in a 3D simulated world,
learning from a progress-based racing reward with lap timing.

> **Status:** Pipeline working end to end. Circuits validate and close, the environment
> runs in MetaDrive, and a reference-line follower completes a full lap of Suzuka in
> 217 s. No trained policy yet — training is left to you.

---

## Why MetaDrive

The previous version of this project used Gymnasium's `CarRacing-v3`, a 2D Box2D
toy where the reward is "visit tiles quickly". That teaches generic racing; it has no
notion of a racing line, lap time, or tire behaviour, and its top-down 84x84 camera is
about as far from a driver's view as you can get.

MetaDrive is a 3D simulator built on Panda3D with Bullet rigid-body physics, and it
composes road networks out of *blocks* — which is exactly the vocabulary a circuit is
made of:

```text
Straight(length)
Curve(length, radius, angle, dir)   # an arc, then a straight run-off
```

So a circuit becomes data rather than code, and the same blocks that build the road
also build a reference line for measuring progress.

### Requirements this imposes

- **Python 3.11.** MetaDrive declares `requires_python < 3.12`, so 3.11 is the newest
  supported interpreter.
- **No GPU required.** Training runs on CPU. See [Performance](#performance).

---

## Installation

```bash
git clone https://github.com/santhosh220z/Learn-to-drive-with-RL.git
cd Learn-to-drive-with-RL

uv venv .venv --python 3.11          # or: python -m venv .venv
uv pip install --python .venv\Scripts\python.exe -e .
```

**Optional, strongly recommended** — a CUDA build of PyTorch. The default PyPI wheel on
Windows is CPU-only, and a rendered 3D observation is expensive:

```bash
uv pip install --python .venv\Scripts\python.exe torch --index-url https://download.pytorch.org/whl/cu124
uv pip install --python .venv\Scripts\python.exe stable-baselines3 tensorboard
```

Verify:

```bash
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m drive_rl.train --preset smoke --dry-run
```

---

## Usage

### Train

Training never starts implicitly. Every run prints a plan, **measures the environment's
real throughput**, converts it into an estimated wall-clock time, and asks before
committing — a run estimated at 2.5 hours that is really going to take two days is
worth catching up front.

```bash
# check the pipeline without training anything
python -m drive_rl.train --preset smoke --dry-run

# the usual first real run
python -m drive_rl.train --preset balanced

# 3D camera observations (much slower per step)
python -m drive_rl.train --preset camera

# override anything
python -m drive_rl.train --preset balanced --circuit silverstone --n-envs 4
```

| Preset | Steps | Obs | Intended use |
| --- | --- | --- | --- |
| `smoke` | 10k | state | Pipeline check. Will not drive. |
| `fast` | 300k | state | Cheap run; usually laps, slowly. |
| `balanced` | 1.5M | state | Default baseline on one circuit. |
| `camera` | 1M | camera | 3D camera observations. |
| `quality` | 4M | state | Long run; resume from a checkpoint. |

Outputs land in `models/<run>/`: `final_model.zip`, `best_eval/best_model.zip`,
TensorBoard logs under `tb/`, and a `config_snapshot.json` recording the exact settings
and the measured throughput.

### Evaluate

```bash
python -m drive_rl.evaluate --model models/<run>/best_eval/best_model.zip --episodes 20
```

Reports lap times alongside **off-track percentage and worst line deviation**. On a
fixed circuit a policy can turn a fast lap by cutting the run-off, so lap time alone is
not a sufficient measure. Circuit and observation mode are read from the run's
`config_snapshot.json` automatically.

### Watch

```bash
python -m drive_rl.watch --run-dir models/<run>
```

Polls the run directory and hot-reloads the best checkpoint whenever one is saved, so you
can watch the policy improve without stopping training. Use `--model <path> --loop` to
replay a single model.

### Record

```bash
python -m drive_rl.record_video --model models/<run>/best_eval/best_model.zip
```

---

## Circuits

Circuits live as data in `src/drive_rl/circuits.py`, not as code.

| Circuit | Length | Corners | Real circuit |
| --- | --- | --- | --- |
| `suzuka` | 4546 m | 16 | 5807 m, 18 turns |
| `silverstone` | 3641 m | 12 | 5891 m, 18 turns |
| `monaco` | 2221 m | 15 | 3337 m, 19 turns |

**A circuit must close on itself** — the accumulated turn has to be one full revolution
and the path has to return to its start, or the road network's end never meets its
beginning and laps become impossible. MetaDrive builds a broken map *silently*, so
`validate_circuit` checks both conditions and `tests/test_circuits.py` asserts them for
every shipped circuit.

Authoring is assisted by `tools/close_circuit.py`, which closes a draft numerically
instead of by hand:

```bash
python tools/close_circuit.py --list
python tools/close_circuit.py suzuka --emit
```

### Fidelity limits

These are deliberate approximations, not surveyed centrelines:

- **Flat.** MetaDrive blocks are 2D, so Suzuka's ~40 m of elevation change is absent.
- **Constant-radius corners.** Real corners vary in radius and compound; an "S" is
  written as two opposite arcs.
- **~20–30% shorter** than the real circuits, with fewer corners, as a consequence of the
  above plus the closure solve.
- **No crossover or kerbs.** Suzuka's bridge over the back straight cannot be expressed
  by a flat non-intersecting block chain, and track edges are guardrails.

---

## How the reward works

Reward is expressed in **lap units**: a clean lap is worth `progress_reward`.

That framing has a consequence worth stating, because getting it wrong silently ruins
training. A step covers about 2 m of a 4546 m lap, so the progress reward *per step* is
only ~0.005. Penalties are charged per step, so they must be scaled to match — an
off-road penalty in single digits would be a thousand times the progress signal and
would dominate every gradient.

```text
progress_reward   10.0    a full lap, prorated each step
lateral_penalty   0.02    squared lateral offset, capped at 1.5 track-half-widths
offroad_penalty   0.20    per step off the racing surface
lap_bonus         5.0     for crossing the start/finish line
```

Laps are detected by the reference line's arc length wrapping past the end of the loop.
Progress and lateral offset come from projecting the car onto the centreline built from
the *same* geometry that built the road, so they stay well-defined across block
boundaries and do not depend on MetaDrive's internal lane identifiers.

---

## Performance

Measured on the development machine (4 cores / 8 threads, RTX 3050 6 GB, ~12 GB RAM):

| Configuration | Throughput |
| --- | --- |
| State obs (lidar), headless | ~1135 env-steps/s |
| 3D camera obs 84x84, offscreen | ~55 env-steps/s |
| 3D camera obs 120x120, offscreen | ~55 env-steps/s |

Resolution barely changes throughput, so the cost is per-step overhead rather than pixels.

Two constraints shape how far this can be pushed:

- **MetaDrive allows one environment instance per process**, so parallel environments
  are separate processes and parallelism is capped at your core count.
- **One env step is 0.1 s** (`decision_repeat=5` x `physics_world_step_size=0.02`), so a
  Suzuka lap is ~820 steps.

Always trust the ETA printed by `train.py` — it is measured on the environment it just
built, not estimated from these numbers.

---

## Project structure

```text
src/drive_rl/
├── circuits.py        Circuit specs, closure validation, reference line, map builder
├── racing_env.py      RacingEnv: lap counting, racing reward, SB3 VecEnv factory
├── train.py           PPO CLI with presets, --dry-run, measured ETA
├── evaluate.py        Deterministic eval with lap times and line deviation
├── watch.py           3D live viewer, hot-reloads the best checkpoint
├── record_video.py    Offscreen 3D video capture
├── callbacks.py       SB3 callback composition
└── utils.py           Seeding, device selection, config snapshots
tools/
└── close_circuit.py   Offline circuit-closing authoring tool
tests/
├── test_circuits.py       Geometry, closure, reference line
└── test_racing_env.py     Env contracts, camera layout, reward scaling
```

---

## Development

```bash
pytest -q
ruff check src tests tools
black src tests tools
```

`tests/test_racing_env.py` builds real MetaDrive environments and is skipped when
MetaDrive is unavailable. The geometry tests always run.

## License

MIT — see [LICENSE](LICENSE).