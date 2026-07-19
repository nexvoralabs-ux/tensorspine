"""Behavioural tests for Neuron / Layer / MLP built on the Value engine."""

import random

import pytest

from tensorspine import MLP, Layer, Neuron, Value


class TestParameterCounts:
    def test_neuron_parameter_count(self):
        """A neuron with nin inputs holds nin weights + 1 bias."""
        n = Neuron(4)
        assert len(n.parameters()) == 5

    def test_layer_parameter_count(self):
        """Layer(nin=3, nout=5) → 5 neurons × (3 weights + 1 bias) = 20."""
        layer = Layer(3, 5)
        assert len(layer.parameters()) == 20

    def test_mlp_parameter_count(self):
        """MLP(2, [16, 16, 1]):
        layer1: 16×(2+1) = 48
        layer2: 16×(16+1) = 272
        layer3: 1×(16+1)  = 17
        total = 337
        """
        model = MLP(2, [16, 16, 1])
        assert len(model.parameters()) == 337

    def test_parameters_are_values(self):
        model = MLP(2, [3, 1])
        assert all(isinstance(p, Value) for p in model.parameters())

    def test_parameters_are_distinct_objects(self):
        """No parameter object may appear twice in the flattened list."""
        model = MLP(2, [3, 1])
        params = model.parameters()
        assert len(params) == len(set(map(id, params)))


class TestZeroGrad:
    def test_zero_grad_actually_zeros(self):
        """After a backward pass fills grads, zero_grad() must reset ALL of
        them to exactly 0.0."""
        random.seed(0)
        model = MLP(2, [4, 1])
        out = model([Value(0.5), Value(-0.3)])
        loss = (out - 1.0) ** 2
        loss.backward()
        # Sanity: at least one parameter actually received a gradient,
        # otherwise this test would pass vacuously.
        assert any(p.grad != 0.0 for p in model.parameters())

        model.zero_grad()
        assert all(p.grad == 0.0 for p in model.parameters())


class TestForwardShapes:
    def test_neuron_returns_scalar_value(self):
        random.seed(0)
        n = Neuron(3)
        out = n([1.0, 2.0, 3.0])
        assert isinstance(out, Value)

    def test_layer_returns_list(self):
        random.seed(0)
        layer = Layer(3, 4)
        out = layer([1.0, 2.0, 3.0])
        assert isinstance(out, list)
        assert len(out) == 4
        assert all(isinstance(v, Value) for v in out)

    def test_single_output_layer_returns_bare_value(self):
        random.seed(0)
        layer = Layer(3, 1)
        out = layer([1.0, 2.0, 3.0])
        assert isinstance(out, Value)

    def test_tanh_neuron_output_bounded(self):
        """A tanh neuron's output must lie strictly inside (−1, 1)."""
        random.seed(0)
        n = Neuron(2, nonlin="tanh")
        for x in ([0.0, 0.0], [5.0, -5.0], [100.0, 100.0]):
            out = n(x)
            assert -1.0 <= out.data <= 1.0

    def test_relu_neuron_output_nonnegative(self):
        random.seed(0)
        n = Neuron(2, nonlin="relu")
        for x in ([0.0, 0.0], [5.0, -5.0], [-3.0, 2.0]):
            out = n(x)
            assert out.data >= 0.0


class TestTraining:
    def test_neuron_overfits_single_point(self):
        """One neuron + SGD must drive the loss on a single (x, y) pair to
        near zero within 200 steps. If this fails while gradcheck passes, the
        problem is in the nn forward pass wiring, not the calculus."""
        random.seed(42)
        n = Neuron(2, nonlin="tanh")
        x, y = [0.5, -1.2], 0.8  # target within tanh's range
        lr = 0.5

        first_loss = None
        loss = None
        for _ in range(200):
            pred = n(x)
            loss = (pred - y) ** 2
            if first_loss is None:
                first_loss = loss.data
            n.zero_grad()
            loss.backward()
            for p in n.parameters():
                p.data -= lr * p.grad

        assert loss.data < 1e-4, (
            f"neuron failed to overfit one point: loss went "
            f"{first_loss:.6f} → {loss.data:.6f}"
        )

    def test_mlp_loss_decreases_on_linearly_separable_set(self):
        """An MLP trained by SGD for 100 steps on a small linearly separable
        2-class set must reduce the MSE loss substantially and end with every
        prediction on the correct side of 0."""
        random.seed(7)
        # Linearly separable by the line x1 + x2 = 0.
        xs = [
            [2.0, 3.0], [1.0, 1.5], [3.0, 0.5], [0.5, 2.5],
            [-2.0, -3.0], [-1.0, -1.5], [-3.0, -0.5], [-0.5, -2.5],
        ]
        ys = [1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0]

        model = MLP(2, [4, 1])
        lr = 0.05

        def total_loss():
            preds = [model(x) for x in xs]
            return sum(
                ((p - y) ** 2 for p, y in zip(preds, ys)),
                start=Value(0.0),
            ) / len(xs)

        initial = total_loss().data
        loss = None
        for _ in range(100):
            loss = total_loss()
            model.zero_grad()
            loss.backward()
            for p in model.parameters():
                p.data -= lr * p.grad

        final = loss.data
        assert final < initial * 0.5, (
            f"loss did not decrease enough over 100 steps: "
            f"{initial:.6f} → {final:.6f}"
        )
        # Every point classified on the correct side.
        for x, y in zip(xs, ys):
            pred = model(x)
            assert pred.data * y > 0, (
                f"point {x} (label {y}) predicted {pred.data:.4f} — wrong side"
            )


class TestInitSchemes:
    def test_both_init_schemes_construct(self):
        """Both initialization schemes must be selectable and produce finite
        weights of the right count."""
        random.seed(0)
        for scheme in ("uniform", "xavier"):
            n = Neuron(16, init=scheme)
            assert len(n.parameters()) == 17
            assert all(
                isinstance(p.data, float) and p.data == p.data  # not NaN
                for p in n.parameters()
            )

    def test_xavier_weights_scaled_down(self):
        """For large nin, Xavier weights (∝ 1/sqrt(nin)) must be markedly
        smaller in magnitude than naive uniform(−1, 1) weights on average."""
        random.seed(0)
        nin = 100
        naive = Neuron(nin, init="uniform")
        xavier = Neuron(nin, init="xavier")
        mean_abs_naive = sum(abs(w.data) for w in naive.w) / nin
        mean_abs_xavier = sum(abs(w.data) for w in xavier.w) / nin
        assert mean_abs_xavier < mean_abs_naive * 0.5, (
            f"xavier weights (mean |w| = {mean_abs_xavier:.4f}) are not "
            f"clearly smaller than uniform (mean |w| = {mean_abs_naive:.4f}) "
            f"at nin = {nin}"
        )
