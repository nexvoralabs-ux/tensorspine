"""Experiment 03 — what zero_grad() is for.

Trains the SAME model on the SAME data twice: once calling zero_grad() before
each backward pass, and once deliberately NOT calling it. Overlays the two loss
curves and saves experiments/out/zero_grad_comparison.png, then prints a short
explanation of what the divergence demonstrates.

Run:  python experiments/03_no_zero_grad.py
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
STEPS = 100
LR = 0.02
SEED = 2024


def make_data():
    X, y01 = make_moons(n_samples=N_SAMPLES, noise=NOISE, random_state=SEED)
    return X, y01 * 2.0 - 1.0


def batch_loss(model, X, y):
    preds = [model(list(map(float, xi))) for xi in X]
    total = sum(((p - float(yi)) ** 2 for p, yi in zip(preds, y)), start=Value(0.0))
    return total / len(X)


def train(zero_grad: bool):
    """Return the per-step loss history for a fresh model.

    The two runs use the same seed, so the ONLY difference is whether
    zero_grad() is called each step.
    """
    random.seed(SEED)
    np.random.seed(SEED)
    X, y = make_data()
    model = MLP(2, [16, 16, 1], nonlin="tanh", init="xavier")

    history = []
    for _ in range(STEPS):
        loss = batch_loss(model, X, y)
        if zero_grad:
            model.zero_grad()
        loss.backward()
        sgd_step(model, LR)
        history.append(loss.data)
    return history


def main():
    print("training WITH zero_grad()...")
    with_zg = train(zero_grad=True)
    print("training WITHOUT zero_grad()...")
    without_zg = train(zero_grad=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(with_zg, label="with zero_grad()", color="tab:green", lw=2)
    ax.plot(without_zg, label="without zero_grad()", color="tab:red", lw=2)
    ax.set_xlabel("step")
    ax.set_ylabel("MSE loss")
    ax.set_title("Effect of zero_grad() on training")
    ax.legend()
    ax.grid(alpha=0.3)

    out_path = OUT_DIR / "zero_grad_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"saved {out_path}")

    print("\n" + "=" * 68)
    print("WHAT THIS SHOWS")
    print("=" * 68)
    print(
        "Gradients ACCUMULATE (+=) across backward() calls; they are not reset\n"
        "automatically. zero_grad() clears them before each step.\n\n"
        "WITHOUT zero_grad(), step k's parameter gradients still carry the\n"
        "summed gradients of steps 0..k−1. Each SGD update therefore uses a\n"
        "gradient that is effectively the running SUM of all past gradients —\n"
        "an ever-growing, stale, wrong-magnitude step. The loss curve becomes\n"
        "erratic and/or diverges instead of descending smoothly.\n\n"
        "This is the training-loop consequence of the SAME accumulation\n"
        "semantics that make shared-node gradients correct inside a single\n"
        "backward pass. The += that is essential within one backward() is\n"
        "exactly why you must zero_grad() between successive backward() calls."
    )
    print(f"\nfinal loss  with zero_grad(): {with_zg[-1]:.6f}")
    print(f"final loss  without        : {without_zg[-1]:.6f}")


if __name__ == "__main__":
    main()
