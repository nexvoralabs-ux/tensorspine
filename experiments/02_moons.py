"""Experiment 02 — train an MLP on the sklearn two-moons dataset.

Generates the moons dataset, trains MLP(2, [16, 16, 1]) with SGD on an MSE
loss against ±1 labels, prints the loss curve, and saves a decision-boundary
plot to experiments/out/moons_boundary.png.

Run:  python experiments/02_moons.py
"""

import random

import matplotlib

matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons

from _common import OUT_DIR, sgd_step

from tensorspine import MLP, Value

N_SAMPLES = 100
NOISE = 0.1
STEPS = 150
LR = 0.05
BOUNDARY_RES = 60  # grid resolution for the decision-boundary plot


def batch_loss(model, X, y):
    """Mean squared error of the model over the whole dataset."""
    preds = [model(list(map(float, xi))) for xi in X]
    total = sum(
        ((p - float(yi)) ** 2 for p, yi in zip(preds, y)), start=Value(0.0)
    )
    return total / len(X), preds


def accuracy(preds, y):
    return sum(1 for p, yi in zip(preds, y) if p.data * yi > 0) / len(y)


def main():
    random.seed(1337)
    np.random.seed(1337)

    X, y01 = make_moons(n_samples=N_SAMPLES, noise=NOISE, random_state=1337)
    y = y01 * 2.0 - 1.0  # labels in {−1, +1} for a tanh-flavored net

    model = MLP(2, [16, 16, 1], nonlin="tanh", init="xavier")
    print(f"model: {sum(1 for _ in model.parameters())} parameters")
    print(f"{'step':>5}  {'loss':>10}  {'acc':>6}")
    print("-" * 26)

    for step in range(STEPS):
        loss, preds = batch_loss(model, X, y)
        model.zero_grad()
        loss.backward()
        sgd_step(model, LR)

        if step % 10 == 0 or step == STEPS - 1:
            print(f"{step:>5}  {loss.data:>10.6f}  {accuracy(preds, y):>6.1%}")

    # ---------------------------------------------------------------- plot --
    print("rendering decision boundary...")
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, BOUNDARY_RES),
        np.linspace(y_min, y_max, BOUNDARY_RES),
    )
    zz = np.array(
        [
            model([float(a), float(b)]).data
            for a, b in zip(xx.ravel(), yy.ravel())
        ]
    ).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.contourf(xx, yy, zz > 0, alpha=0.25, levels=1, cmap="coolwarm")
    ax.contour(xx, yy, zz, levels=[0.0], colors="k", linewidths=1.5)
    ax.scatter(
        X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolors="k", s=30
    )
    ax.set_title("tensorspine MLP(2, [16, 16, 1]) — moons decision boundary")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")

    out_path = OUT_DIR / "moons_boundary.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
