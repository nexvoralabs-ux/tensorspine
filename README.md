# tensorspine

A scalar reverse-mode automatic differentiation engine, written from scratch to
understand backpropagation.

[![CI](https://github.com/advik/tensorspine/actions/workflows/test.yml/badge.svg)](https://github.com/advik/tensorspine/actions/workflows/test.yml)

> **Status: in progress — implementing by hand, one gradient rule at a time.**
> Tests fail by design until the engine is complete.
> **Stubs remaining: 19 / 19**  (14 core gradient methods + 4 reflected
> operators + 1 topological-sort helper)

## Why this exists

Backpropagation is the chain rule applied to a directed acyclic graph in reverse
topological order; everything else is bookkeeping. This engine is a few hundred
lines of pure Python doing exactly what PyTorch's autograd does, at a scale small
enough to hold entirely in your head. The point is not the code — it is being
able to derive every gradient rule and the reverse traversal from first
principles, without looking them up.

## What's implemented vs. by hand

| Generated scaffolding (complete) | Written by hand (the exercise) |
| --- | --- |
| Test suite (`tests/`) | Every local gradient rule |
| Experiments (`experiments/`) | `backward()` + topological sort |
| Graphviz DAG visualizer (`viz.py`) | `Neuron` / `Layer` / `MLP` forward passes |
| `parameters()`, `zero_grad()` | weight initialization |

The stub inventory lives in `CLAUDE.md`; `make stubs` lists what remains.

## Quickstart

```bash
git clone <repo> && cd tensorspine
make setup
make test        # fails until stubs are done — expected
make stubs       # see what's left
```

## Design constraints

- The engine imports only `math` and `random`. Nothing else.
- Scalar-valued: every node holds one float, no tensors.
- Readability over speed.

`numpy`, `matplotlib`, `scikit-learn`, and `torch` appear only in the tests and
experiments — never in the engine. The empty `dependencies` list in
`pyproject.toml` enforces that this stays true.

## The core idea

For a function with `n` inputs and `m` outputs, forward-mode autodiff costs scale
with `n` (one pass per input) and reverse-mode with `m` (one pass per output).
Reverse-mode propagates a gradient *backward* from each output; forward-mode
propagates a derivative *forward* from each input.

In machine learning the loss is a single scalar (`m = 1`) and the parameters
number in the millions (`n` huge). Reverse-mode gets every one of those millions
of partial derivatives in a single backward pass. That asymmetry is why training
deep networks is computationally feasible at all.

## Verification

Correctness is established two independent ways:

- **Numerical gradient checking** — every analytic gradient is compared against a
  central-difference estimate `(f(x+h) − f(x−h)) / 2h` and must agree to a
  relative error below `1e-6` (`tests/test_gradcheck.py`).
- **PyTorch cross-check** — an identical small network is built in PyTorch, the
  hand-written engine's weights are copied across, and every gradient — weights,
  biases, and inputs — must match PyTorch's to `1e-6` (`tests/test_vs_pytorch.py`).

## Experiments

- `01_single_neuron.py` — overfit one point; the smallest end-to-end training loop.
- `02_moons.py` — train an MLP on the sklearn moons set; plot the decision boundary.
- `03_no_zero_grad.py` — training with vs. without `zero_grad()`; shows why
  gradients must be reset between steps.
- `04_init_comparison.py` — naive `uniform(-1, 1)` vs. Xavier initialization,
  convergence overlaid.
- `05_deep_chain.py` — 50 stacked `tanh` ops; measures the vanishing gradient
  numerically.

## Learning-mode setup

This repo is configured to *teach* rather than autocomplete. `CLAUDE.md` puts the
AI assistant in a tutor role: it explains the mathematics and diagnoses failing
tests, but refuses to write the gradient code. `.claude/settings.json` goes
further and mechanically write-blocks `tensorspine/engine.py` and
`tensorspine/nn.py`, so the two files that must be earned cannot be edited by the
tool even on request. The friction is the point — see `notes/shortcuts.md`, which
logs any time the honest path was skipped.

## Credit

In the lineage of Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd)
as a pedagogical exercise, independently implemented.
