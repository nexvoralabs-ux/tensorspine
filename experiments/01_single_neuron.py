"""Experiment 01 — overfit a single data point with one neuron.

The smallest possible end-to-end training loop: one neuron, one (x, y) pair,
squared-error loss, vanilla SGD. Prints the loss at every step. If your engine
and neuron are correct, the loss collapses toward zero within ~100 steps.

Run:  python experiments/01_single_neuron.py
"""

import random

from _common import sgd_step  # noqa: F401  (also fixes sys.path)

from tensorspine import Neuron

STEPS = 100
LR = 0.5


def main():
    random.seed(42)
    neuron = Neuron(3, nonlin="tanh")

    x = [1.0, -2.0, 3.0]
    y = 0.5  # target, comfortably inside tanh's (−1, 1) range

    print(f"Overfitting one point: x = {x}, target y = {y}")
    print(f"{'step':>5}  {'pred':>10}  {'loss':>12}")
    print("-" * 32)

    for step in range(STEPS):
        pred = neuron(x)
        loss = (pred - y) ** 2

        neuron.zero_grad()
        loss.backward()
        sgd_step(neuron, LR)

        print(f"{step:>5}  {pred.data:>10.6f}  {loss.data:>12.8f}")

    final_pred = neuron(x)
    print("-" * 32)
    print(f"final prediction: {final_pred.data:.6f}  (target {y})")
    print(f"final loss:       {((final_pred - y) ** 2).data:.10f}")


if __name__ == "__main__":
    main()
