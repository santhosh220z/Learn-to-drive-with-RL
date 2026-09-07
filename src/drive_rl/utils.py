"""Utility helpers: seeding, device selection, and config I/O."""

from __future__ import annotations

import json
import random
import warnings
from pathlib import Path
from typing import Any

import numpy as np


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, and (if available) PyTorch RNGs.

    Gymnasium / SB3 have their own seeding paths; call this once at the top of
    any script so non-env randomness is also deterministic.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Make cudnn deterministic when possible. May slow training a little.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        warnings.warn("PyTorch not installed; skipping torch seeding.", stacklevel=2)


def select_device(device: str = "auto") -> str:
    """Resolve a torch device string. Returns 'cpu' if torch is missing."""
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def save_config_snapshot(config: dict[str, Any], path: Path) -> None:
    """Persist a JSON snapshot of training configuration alongside the model."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True, default=str)
