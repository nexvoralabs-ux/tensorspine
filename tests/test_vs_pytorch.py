"""Capstone: gradient agreement with PyTorch to 1e-6.

We build an identical tiny MLP in PyTorch, copy YOUR engine's weights across,
run both forward+backward passes on the same data, and demand every single
gradient — every weight, every bias, every input — agrees to 1e-6.

Skips cleanly if torch is not installed.
"""

import random

import pytest

torch = pytest.importorskip(
    "torch",
    reason="PyTorch is not installed — install the dev extras "
    "(pip install -e '.[dev]') to run the cross-check suite.",
)

from tensorspine import MLP, Value  # noqa: E402

TOL = 1e-6


def assert_close(mine, theirs, label):
    assert mine == pytest.approx(theirs, rel=TOL, abs=TOL), (
        f"{label}: tensorspine = {mine!r}  vs  pytorch = {theirs!r}  "
        f"(diff = {abs(mine - theirs):.3e})"
    )


# --------------------------------------------------------------------------- #
# Scalar expression cross-checks — small, surgical, easy to debug.
# --------------------------------------------------------------------------- #
class TestScalarExpressions:
    def test_mixed_expression(self):
        """z = tanh(a·b + b³) · exp(a/2) − b, at a=0.7, b=−1.1."""
        a_v, b_v = Value(0.7), Value(-1.1)
        z_v = (a_v * b_v + b_v ** 3).tanh() * (a_v / 2.0).exp() - b_v
        z_v.backward()

        a_t = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
        b_t = torch.tensor(-1.1, dtype=torch.float64, requires_grad=True)
        z_t = torch.tanh(a_t * b_t + b_t ** 3) * torch.exp(a_t / 2.0) - b_t
        z_t.backward()

        assert_close(z_v.data, z_t.item(), "forward value")
        assert_close(a_v.grad, a_t.grad.item(), "grad of a")
        assert_close(b_v.grad, b_t.grad.item(), "grad of b")

    def test_fanout_expression(self):
        """z = (x·y + x)·relu(x − y), a shared-node expression, x=1.3, y=0.4."""
        x_v, y_v = Value(1.3), Value(0.4)
        z_v = (x_v * y_v + x_v) * (x_v - y_v).relu()
        z_v.backward()

        x_t = torch.tensor(1.3, dtype=torch.float64, requires_grad=True)
        y_t = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
        z_t = (x_t * y_t + x_t) * torch.relu(x_t - y_t)
        z_t.backward()

        assert_close(z_v.data, z_t.item(), "forward value")
        assert_close(x_v.grad, x_t.grad.item(), "grad of x")
        assert_close(y_v.grad, y_t.grad.item(), "grad of y")


# --------------------------------------------------------------------------- #
# The MLP cross-check.
# --------------------------------------------------------------------------- #
def build_matched_models(nin, hidden, seed=0):
    """Build a tensorspine MLP and an identical torch model sharing weights.

    Architecture: Linear(nin→hidden) → tanh → Linear(hidden→1), matching
    MLP(nin, [hidden, 1]) whose final layer is linear by construction.

    The tensorspine model is created first (with its own random init); its
    weights are then COPIED into the torch model, so both start identical.
    """
    random.seed(seed)
    mine = MLP(nin, [hidden, 1], nonlin="tanh")

    theirs = torch.nn.Sequential(
        torch.nn.Linear(nin, hidden),
        torch.nn.Tanh(),
        torch.nn.Linear(hidden, 1),
    ).double()

    with torch.no_grad():
        for layer_idx, torch_idx in ((0, 0), (1, 2)):
            my_layer = mine.layers[layer_idx]
            t_linear = theirs[torch_idx]
            for i, neuron in enumerate(my_layer.neurons):
                for j, w in enumerate(neuron.w):
                    t_linear.weight[i, j] = w.data
                t_linear.bias[i] = neuron.b.data

    return mine, theirs


def collect_grads(mine, theirs):
    """Yield (label, my_grad, their_grad) for every parameter, aligned."""
    for layer_idx, torch_idx in ((0, 0), (1, 2)):
        my_layer = mine.layers[layer_idx]
        t_linear = theirs[torch_idx]
        for i, neuron in enumerate(my_layer.neurons):
            for j, w in enumerate(neuron.w):
                yield (
                    f"layer{layer_idx}.neuron{i}.w{j}",
                    w.grad,
                    t_linear.weight.grad[i, j].item(),
                )
            yield (
                f"layer{layer_idx}.neuron{i}.b",
                neuron.b.grad,
                t_linear.bias.grad[i].item(),
            )


class TestMLPCrossCheck:
    def test_forward_agreement(self):
        """Same weights, same input → same output, before any training."""
        mine, theirs = build_matched_models(nin=2, hidden=3, seed=0)
        x = [0.5, -1.2]

        my_out = mine(x)
        t_out = theirs(torch.tensor(x, dtype=torch.float64))

        assert_close(my_out.data, t_out.item(), "MLP forward output")

    def test_backward_agreement_single_sample(self):
        """MSE on one sample: every parameter gradient agrees to 1e-6."""
        mine, theirs = build_matched_models(nin=2, hidden=3, seed=0)
        x, y = [0.5, -1.2], 0.7

        my_loss = (mine(x) - y) ** 2
        my_loss.backward()

        t_out = theirs(torch.tensor(x, dtype=torch.float64))
        t_loss = (t_out - y) ** 2
        t_loss.backward()

        assert_close(my_loss.data, t_loss.item(), "loss value")
        checked = 0
        for label, mg, tg in collect_grads(mine, theirs):
            assert_close(mg, tg, label)
            checked += 1
        assert checked == len(mine.parameters())

    def test_backward_agreement_batch(self):
        """Mean MSE over a small batch — gradients sum over samples exactly as
        PyTorch's do."""
        mine, theirs = build_matched_models(nin=3, hidden=4, seed=1)
        xs = [
            [0.5, -1.2, 0.3],
            [-0.7, 0.9, 1.1],
            [1.5, 0.2, -0.4],
            [-1.0, -0.5, 0.8],
        ]
        ys = [0.7, -0.3, 0.1, 0.9]

        my_loss = sum(
            ((mine(x) - y) ** 2 for x, y in zip(xs, ys)), start=Value(0.0)
        ) / len(xs)
        my_loss.backward()

        t_x = torch.tensor(xs, dtype=torch.float64)
        t_y = torch.tensor(ys, dtype=torch.float64).unsqueeze(1)
        t_loss = ((theirs(t_x) - t_y) ** 2).mean()
        t_loss.backward()

        assert_close(my_loss.data, t_loss.item(), "batch loss value")
        for label, mg, tg in collect_grads(mine, theirs):
            assert_close(mg, tg, label)

    def test_input_gradients_agree(self):
        """Gradients w.r.t. the INPUTS (not just weights) also agree — this
        exercises paths through the graph that parameter grads can miss."""
        mine, theirs = build_matched_models(nin=2, hidden=3, seed=2)
        x_vals = [0.4, -0.9]

        my_inputs = [Value(v) for v in x_vals]
        my_loss = (mine(my_inputs) - 0.5) ** 2
        my_loss.backward()

        t_in = torch.tensor(x_vals, dtype=torch.float64, requires_grad=True)
        t_loss = (theirs(t_in) - 0.5) ** 2
        t_loss.backward()

        for i, mv in enumerate(my_inputs):
            assert_close(mv.grad, t_in.grad[i].item(), f"input[{i}]")
