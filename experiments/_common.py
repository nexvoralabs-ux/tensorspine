"""Shared helpers for the experiment scripts.

Makes the scripts runnable both from an installed package (pip install -e .)
and directly from a fresh clone (by putting the repo root on sys.path), and
provides the output directory for generated PNGs.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

OUT_DIR = Path(__file__).resolve().parent / "out"
OUT_DIR.mkdir(exist_ok=True)


def sgd_step(model, lr):
    """One vanilla SGD update over all parameters of a tensorspine module."""
    for p in model.parameters():
        p.data -= lr * p.grad
