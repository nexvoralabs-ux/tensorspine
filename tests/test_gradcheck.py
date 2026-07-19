"""Numerical gradient checking via central differences.

For every operation and a batch of random composite expressions, we compare the
engine's analytic gradients (from .backward()) against the central-difference
estimate

    f'(x) ≈ (f(x + h) − f(x − h)) / (2h)

computed by re-running the FORWARD pass with perturbed inputs. Central
differences have O(h²) truncation error, so with h = 1e-5 an agreement to
relative error < 1e-6 is expected for any correct implementation.

The report helper says exactly WHICH input mismatched and by how much —
debuggability matters more than terseness here.
"""

import random

import pytest

from tensorspine import Value

H = 1e-5
TOL = 1e-6


def numerical_grad(f, x, h=1e-5):
    """Central-difference derivative of a scalar function of one float."""
    return (f(x + h) - f(x - h)) / (2 * h)


def gradcheck(build, xs, h=H, tol=TOL):
    """Check analytic vs numerical gradients for a multi-input expression.

    Args:
        build: callable taking len(xs) Value arguments and returning the output
               Value (the expression under test).
        xs:    list of float input points.
        h:     central-difference step.
        tol:   maximum allowed relative error per input.

    Fails the test with a detailed per-input report if any input's analytic
    gradient disagrees with the numerical estimate.
    """
    # Analytic gradients from the engine's backward pass.
    leaves = [Value(x) for x in xs]
    out = build(*leaves)
    out.backward()
    analytic = [leaf.grad for leaf in leaves]

    # Numerical gradient per input, holding the others fixed.
    failures = []
    for i in range(len(xs)):

        def f_of_xi(xi, i=i):
            perturbed = list(xs)
            perturbed[i] = xi
            return build(*[Value(p) for p in perturbed]).data

        numeric = numerical_grad(f_of_xi, xs[i], h)
        scale = max(1.0, abs(analytic[i]), abs(numeric))
        rel_err = abs(analytic[i] - numeric) / scale
        if rel_err > tol:
            failures.append(
                f"  input[{i}] (x = {xs[i]!r}):\n"
                f"    analytic  = {analytic[i]!r}\n"
                f"    numerical = {numeric!r}\n"
                f"    rel error = {rel_err:.3e}  (tolerance {tol:.1e})"
            )

    if failures:
        pytest.fail(
            "Gradient check FAILED — analytic (backward) disagrees with "
            "central differences:\n" + "\n".join(failures)
        )


# --------------------------------------------------------------------------- #
# Every primitive and derived operation, one at a time.
# Inputs are chosen away from non-differentiable points (relu at 0) and away
# from invalid domains (fractional powers of negatives, division near 0).
# --------------------------------------------------------------------------- #
SINGLE_OPS = [
    pytest.param(lambda a, b: a + b, [2.0, -3.0], id="add"),
    pytest.param(lambda a, b: a * b, [2.0, -3.0], id="mul"),
    pytest.param(lambda a: a ** 2, [1.7], id="pow2"),
    pytest.param(lambda a: a ** 3, [-1.3], id="pow3"),
    pytest.param(lambda a: a ** -1, [2.5], id="pow_neg1"),
    pytest.param(lambda a: a ** 0.5, [4.2], id="pow_half"),
    pytest.param(lambda a: a.exp(), [0.7], id="exp"),
    pytest.param(lambda a: a.tanh(), [0.4], id="tanh"),
    pytest.param(lambda a: a.relu(), [1.2], id="relu_active"),
    pytest.param(lambda a: a.relu(), [-1.2], id="relu_dead"),
    pytest.param(lambda a: -a, [3.1], id="neg"),
    pytest.param(lambda a, b: a - b, [5.0, 1.5], id="sub"),
    pytest.param(lambda a, b: a / b, [3.0, 2.0], id="div"),
    pytest.param(lambda a: 2.5 + a, [1.0], id="radd"),
    pytest.param(lambda a: 2.5 * a, [1.0], id="rmul"),
    pytest.param(lambda a: 2.5 - a, [1.0], id="rsub"),
    pytest.param(lambda a: 2.5 / a, [1.6], id="rtruediv"),
]


@pytest.mark.parametrize("build,xs", SINGLE_OPS)
def test_single_op(build, xs):
    gradcheck(build, xs)


# --------------------------------------------------------------------------- #
# Composite expressions: hand-written shapes exercising fan-out, deep chains,
# and mixed op types.
# --------------------------------------------------------------------------- #
COMPOSITES = [
    pytest.param(lambda a, b: (a * b + a) * (a - b), [1.3, -0.7], id="fanout_mul_add"),
    pytest.param(lambda a, b: (a + b).tanh() * a, [0.4, 0.2], id="tanh_times_input"),
    pytest.param(lambda a: (a * a * a + a * a + a).tanh(), [0.6], id="cubic_into_tanh"),
    pytest.param(lambda a, b: (a / b + b / a) ** 2, [1.5, 2.5], id="ratio_symmetric_sq"),
    pytest.param(lambda a, b: (a * b).exp() / (1.0 + (a * b).exp()), [0.3, 0.5], id="sigmoid_shape"),
    pytest.param(lambda a: a.tanh().tanh().tanh(), [0.8], id="triple_tanh_chain"),
    pytest.param(lambda a, b: (a.relu() + b.relu()) * (a + b), [1.1, 2.2], id="relu_sum_product"),
    pytest.param(lambda a, b, c: ((a * b + c) ** 2 + (a - c) ** 2).tanh(), [0.3, -0.4, 0.2], id="three_var_quadratics"),
    pytest.param(lambda a, b: (2.0 * a + 3.0 / b - 1.0) * (b - a) ** 3, [0.5, 1.5], id="reflected_mix"),
    pytest.param(lambda a, b, c: (a.exp() + b.tanh()) / (c ** 2 + 1.0), [0.2, 0.7, 1.1], id="exp_tanh_ratio"),
]


@pytest.mark.parametrize("build,xs", COMPOSITES)
def test_composite_expression(build, xs):
    gradcheck(build, xs)


# --------------------------------------------------------------------------- #
# Randomized composites: a fixed-seed generator builds expression trees from
# safe ops, then gradchecks each one. 10 trees × several inputs each.
# --------------------------------------------------------------------------- #
def _random_expr(rng, depth):
    """Build a random expression as a function of two Value inputs.

    Uses only ops that are smooth on all of ℝ (add/sub/mul/tanh/exp-of-tanh)
    so any random input point is a valid gradcheck point.
    """
    if depth == 0:
        return rng.choice(
            [
                lambda a, b: a,
                lambda a, b: b,
                lambda a, b, k=rng.uniform(-2, 2): a * k,
                lambda a, b, k=rng.uniform(-2, 2): b + k,
            ]
        )
    left = _random_expr(rng, depth - 1)
    right = _random_expr(rng, depth - 1)
    combiner = rng.choice(
        [
            lambda l, r: l + r,
            lambda l, r: l - r,
            lambda l, r: l * r,
            lambda l, r: l.tanh() + r,
            lambda l, r: (l * 0.3).exp() * r,
        ]
    )
    return lambda a, b, L=left, R=right, C=combiner: C(L(a, b), R(a, b))


@pytest.mark.parametrize("seed", range(10))
def test_random_composite(seed):
    rng = random.Random(1234 + seed)
    build = _random_expr(rng, depth=3)
    xs = [rng.uniform(-1.5, 1.5), rng.uniform(-1.5, 1.5)]
    gradcheck(build, xs)
