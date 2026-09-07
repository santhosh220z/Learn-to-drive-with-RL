# Learn-to-Drive-with-RL

Train a reinforcement-learning agent to drive on **Gymnasium `CarRacing-v3`** using **PPO** from [Stable-Baselines3](https://stable-baselines3.readthedocs.io/). Vision-based policy (`CnnPolicy`) over stacked 84×84 grayscale frames, with TensorBoard logging, a deterministic eval harness, and a video-recording script for demos.

> **Status**: end-to-end pipeline implemented (env, training, eval, video, tests, docs). Baseline PPO + headline numbers pending a real run on your machine.

---

## Project structure

```text
.
├── pyproject.toml
├── README.md
├── LICENSE
├── .gitignore
├── src/
│   └── drive_rl/
│       ├── __init__.py
│       ├── env.py            # make_env(): gray + resize + frame-stack VecEnv
│       ├── wrappers.py       # Optional: action smoothing, reward shaping, frame skip
│       ├── callbacks.py      # SaveBestModelByReward + EvalCallback composition
│       ├── train.py          # `python -m drive_rl.train`
│       ├── evaluate.py       # `python -m drive_rl.evaluate`
│       ├── record_video.py   # `python -m drive_rl.record_video`
│       └── utils.py          # seeding + device helpers
├── configs/
│   └── default_ppo.yaml      # Hyperparameter preset
├── tests/
│   ├── test_env.py           # VecEnv shape / action / step contracts
│   ├── test_wrappers.py      # Wrapper unit tests
│   └── test_train_smoke.py   # 512-step PPO integration test
├── models/                   # Saved checkpoints (gitignored)
├── videos/                   # Recorded rollouts (gitignored)
└── logs/                     # TensorBoard runs (gitignored)
```

---

## Installation

### Windows (read this first)

`box2d-py` must be built against SWIG on Windows. **Install SWIG first**, then `gymnasium[box2d]`:

```bash
pip install swig
pip install "gymnasium[box2d]"
```

If `box2d-py` build fails, ensure Microsoft Visual C++ build tools are installed (`pip install box2d-py` will compile from sdist otherwise). PyGame is also imported by CarRacing at reset.

### Install the project

```bash
git clone <repo-url> Learn-to-drive-with-RL
cd Learn-to-drive-with-RL
python -m venv .venv
.venv\Scripts\activate     # Windows
# source .venv/bin/activate  # macOS / Linux
pip install -e ".[dev]"
```

Verify:

```bash
python -c "import gymnasium as gym; e = gym.make('CarRacing-v3', continuous=True); print(e.observation_space, e.action_space)"
```

You should see `Box(0, 255, (96, 96, 3), uint8)` and `Box(-1, [+1,+1,+1], (3,), float32)`.

---

## Usage

### Train (smoke test, ~30 seconds on CPU)

```bash
python -m drive_rl.train --timesteps 10000 --n-envs 2
```

### Train (headline baseline)

```bash
python -m drive_rl.train --timesteps 1000000 --n-envs 8 --seed 0
```

Defaults are stored in `configs/default_ppo.yaml` and overridable on the CLI:

```bash
python -m drive_rl.train --learning-rate 1e-4 --n-steps 1024 --ent-coef 0.01
```

Outputs:

- `models/ppo_carracing_<timestamp>/final_model.zip`
- `models/ppo_carracing_<timestamp>/best_eval/best_model.zip` (highest mean eval reward)
- `models/ppo_carracing_<timestamp>/best_model_by_train_reward.zip` (highest mean training reward)
- `models/ppo_carracing_<timestamp>/tb/` — TensorBoard logs
- `models/ppo_carracing_<timestamp>/config_snapshot.json` — exact config used

### Evaluate

```bash
python -m drive_rl.evaluate --model models/ppo_carracing_<ts>/best_eval/best_model.zip --n-episodes 20
```

Writes `eval.json` with per-episode rewards and prints mean ± std.

### Record a demo video

```bash
python -m drive_rl.record_video --model models/ppo_carracing_<ts>/best_eval/best_model.zip --out-dir videos
```

Produces `videos/episode-0.mp4`.

### TensorBoard

```bash
tensorboard --logdir models/ppo_carracing_<ts>/tb
```

---

## How it works

1. **`make_env()`** in [`src/drive_rl/env.py`](src/drive_rl/env.py) builds a vectorized CarRacing-v3 with the standard preprocessing chain:
   - `gym.make("CarRacing-v3", continuous=True)` → `Box(steer, gas, brake) ∈ [-1,1]^3`
   - `GrayscaleObservation` → `(96, 96, 1)`
   - `ResizeObservation` → `(84, 84, 1)`
   - SB3 `VecFrameStack(n_stack=4)` → `(4, 84, 84)` uint8
2. **PPO** with `CnnPolicy` learns a policy over the frame stack.
3. **`EvalCallback`** runs deterministic evaluation every `--eval-freq` steps and saves the best model.
4. **`SaveBestModelByReward`** also tracks the best training reward, in case it diverges.
5. **`evaluate.py`** runs N deterministic episodes and writes a JSON summary.
6. **`record_video.py`** wraps the env in `RecordVideo` and saves an MP4 of one rollout.

### Custom wrappers (opt-in)

`src/drive_rl/wrappers.py` provides three optional wrappers. All are **off by default** so the native reward remains the honest signal:

- `ActionSmoothingWrapper(alpha=0.5)` — EMA over actions to reduce steering jitter.
- `RewardShapingWrapper(steer_penalty=..., forward_bonus=...)` — gentle shaping hooks for ablations.
- `FrameSkipWrapper(skip=4)` — repeat actions for K frames and sum rewards.

Wire them in `env.py` if you want to use them.

---

## Targets & baselines

The conventional **"solved" threshold** for CarRacing is mean reward ≥ 900 over 100 consecutive eval episodes. Realistic numbers for default PPO:

| Configuration | Timesteps | Mean reward (approx) |
| --- | --- | --- |
| Vanilla PPO, default HP | 1M | 150–400 |
| Vanilla PPO, zoo HP | 4M | 700–900 |
| `RecurrentPPO` (sb3-contrib) | 4M | 850+ |

**This README's headline numbers will be filled in after a real run.**

---

## Development

```bash
# Run all tests
pytest -q

# Lint + format
ruff check src tests
black src tests
```

`tests/test_train_smoke.py` runs a 512-step PPO integration test; it is skipped automatically if `gymnasium[box2d]` is not installed.

---

## License

MIT — see [LICENSE](LICENSE).
