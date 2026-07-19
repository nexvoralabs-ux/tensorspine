"""Experiment 05 — vanishing gradients through a deep tanh chain.

Stacks 50 tanh operations in a single chain:

    y = tanh(tanh(tanh(... tanh(x) ...)))     (50 deep)

and measures dy/dx by backpropagating through the whole stack. Because each
tanh contributes a local derivative (1 − tanh²) that is at most 1 and usually
well below it, the product of 50 such factors decays toward zero — this is the
vanishing-gradient problem, shown numerically rather than asserted.

Also plots the gradient magnitude at the input as a function of chain depth,
saved to experiments/out/vanishing_gradient.png.

Run:  python experiments/05_deep_chain.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _common import OUT_DIR

from tensorspine import Value

MAX_DEPTH = 50


def grad_through_chain(x0, depth):
    """Return d(output)/d(input) after `depth` stacked tanh ops at input x0."""
    x = Value(x0)
    y = x
    for _ in range(depth):
        y = y.tanh()
    y.backward()
    return x.grad, y.data


def main():
    x0 = 0.5
    depths = list(range(1, MAX_DEPTH + 1))
    grads = []

    print(f"input x0 = {x0}")
    print(f"{'depth':>6}  {'output':>12}  {'d out/d in':>16}")
    print("-" * 40)
    for d in depths:
        g, out = grad_through_chain(x0, d)
        grads.append(g)
        if d <= 10 or d % 5 == 0:
            print(f"{d:>6}  {out:>12.8f}  {g:>16.3e}")

    print("-" * 40)
    print(
        f"gradient at depth 1:  {grads[0]:.6e}\n"
        f"gradient at depth {MAX_DEPTH}: {grads[-1]:.6e}\n"
        f"shrinkage factor:     {grads[0] / grads[-1]:.3e}×\n\n"
        "Each tanh multiplies the upstream gradient by its local derivative\n"
        "1 − tanh²(·) ≤ 1. Chaining 50 of them multiplies 50 such sub-unit\n"
        "factors together, so the input's gradient decays roughly\n"
        "geometrically with depth — a plain deep tanh stack cannot train its\n"
        "early layers. (This is why residual connections, normalization, and\n"
        "non-saturating activations exist.)"
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(depths, [abs(g) for g in grads], marker="o", ms=3, color="tab:purple")
    ax.set_xlabel("chain depth (number of stacked tanh ops)")
    ax.set_ylabel("|d output / d input|  (log scale)")
    ax.set_title(f"Vanishing gradient through stacked tanh (input x0 = {x0})")
    ax.grid(alpha=0.3, which="both")

    out_path = OUT_DIR / "vanishing_gradient.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
