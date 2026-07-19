"""Experiment 04 — weight initialization: uniform(−1, 1) vs Xavier.

Trains two identical 3-layer MLPs on the moons dataset, differing only in weight
initialization, and overlays their convergence curves. Xavier (∝ 1/sqrt(nin))
generally converges faster and more stably than naive uniform(−1, 1), whose
large weights push tanh units into saturation where gradients vanish.

Saves experiments/out/init_comparison.png.

Run:  python experiments/04_init_comparison.py
"""

import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons

from _common import OUT_DIR, sgd_step

from tensorspine import MLP, Value

N_SAMPLES = 100
NOISE = 0.1
STEPS = 150
LR = 0.05
SEED = 99


def batch_loss(model, X, y):
    preds = [model(list(map(float, xi))) for xi in X]
    total = sum(((p - float(yi)) ** 2 for p, yi in zip(preds, y)), start=Value(0.0))
    return total / len(X)


def train(init_scheme):
    """Train a fresh 3-layer MLP with the given init, return loss history.

    Same seed for both schemes so the data and the random draw ORDER match;
    the only difference is how those draws are scaled into weights.
    """
    random.seed(SEED)
    np.random.seed(SEED)
    X, y01 = make_moons(n_samples=N_SAMPLES, noise=NOISE, random_state=SEED)
    y = y01 * 2.0 - 1.0

    model = MLP(2, [16, 16, 1], nonlin="tanh", init=init_scheme)
    history = []
    for _ in range(STEPS):
        loss = batch_loss(model, X, y)
        model.zero_grad()
        loss.backward()
        sgd_step(model, LR)
        history.append(loss.data)
    return history


def main():
    print("training with uniform(-1, 1) init...")
    uniform_hist = train("uniform")
    print("training with Xavier (1/sqrt(nin)) init...")
    xavier_hist = train("xavier")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(uniform_hist, label="uniform(-1, 1)", color="tab:orange", lw=2)
    ax.plot(xavier_hist, label="Xavier  1/sqrt(nin)", color="tab:blue", lw=2)
    ax.set_xlabel("step")
    ax.set_ylabel("MSE loss")
    ax.set_title("Weight initialization: uniform vs Xavier (3-layer MLP on moons)")
    ax.legend()
    ax.grid(alpha=0.3)

    out_path = OUT_DIR / "init_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"saved {out_path}")

    print(f"\nfinal loss  uniform: {uniform_hist[-1]:.6f}")
    print(f"final loss  xavier : {xavier_hist[-1]:.6f}")
    print(
        "\nXavier keeps the initial pre-activations near zero (where tanh is\n"
        "roughly linear and its gradient is largest), so signal and gradient\n"
        "flow cleanly from step 0. Naive uniform weights make the pre-\n"
        "activations large, saturating tanh units and shrinking their local\n"
        "gradients — slower, noisier early training."
    )


if __name__ == "__main__":
    main()
