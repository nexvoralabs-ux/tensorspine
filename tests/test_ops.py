"""Forward values and backward gradients for every operation.

Every expected number here was computed analytically by hand and hardcoded.
If a test fails, the failure message tells you which op and which direction
(forward vs backward) is wrong.
"""

import pytest

from tensorspine import Value

APPROX = dict(rel=1e-9, abs=1e-12)


# --------------------------------------------------------------------------- #
# Addition: c = a + b   →   ∂c/∂a = 1, ∂c/∂b = 1
# --------------------------------------------------------------------------- #
class TestAdd:
    def test_forward(self):
        c = Value(2.0) + Value(3.0)
        assert c.data == pytest.approx(5.0, **APPROX)

    def test_backward(self):
        a, b = Value(2.0), Value(3.0)
        c = a + b
        c.backward()
        assert a.grad == pytest.approx(1.0, **APPROX)
        assert b.grad == pytest.approx(1.0, **APPROX)

    def test_add_scalar_forward(self):
        c = Value(2.0) + 3.0
        assert c.data == pytest.approx(5.0, **APPROX)

    def test_add_scalar_backward(self):
        a = Value(2.0)
        c = a + 3.0
        c.backward()
        assert a.grad == pytest.approx(1.0, **APPROX)


# --------------------------------------------------------------------------- #
# Multiplication: c = a * b   →   ∂c/∂a = b, ∂c/∂b = a
# --------------------------------------------------------------------------- #
class TestMul:
    def test_forward(self):
        c = Value(2.0) * Value(-3.0)
        assert c.data == pytest.approx(-6.0, **APPROX)

    def test_backward(self):
        a, b = Value(2.0), Value(-3.0)
        c = a * b
        c.backward()
        assert a.grad == pytest.approx(-3.0, **APPROX)  # ∂c/∂a = b
        assert b.grad == pytest.approx(2.0, **APPROX)   # ∂c/∂b = a

    def test_mul_scalar_forward(self):
        c = Value(2.0) * 4.0
        assert c.data == pytest.approx(8.0, **APPROX)

    def test_mul_scalar_backward(self):
        a = Value(2.0)
        c = a * 4.0
        c.backward()
        assert a.grad == pytest.approx(4.0, **APPROX)


# --------------------------------------------------------------------------- #
# Power: c = a ** k   →   ∂c/∂a = k · a^(k−1)
# --------------------------------------------------------------------------- #
class TestPow:
    def test_square_forward(self):
        c = Value(3.0) ** 2
        assert c.data == pytest.approx(9.0, **APPROX)

    def test_square_backward(self):
        a = Value(3.0)
        c = a ** 2
        c.backward()
        assert a.grad == pytest.approx(6.0, **APPROX)  # 2·3^1

    def test_cube_backward(self):
        a = Value(2.0)
        c = a ** 3
        c.backward()
        assert a.grad == pytest.approx(12.0, **APPROX)  # 3·2^2

    def test_negative_exponent_forward(self):
        c = Value(4.0) ** -1
        assert c.data == pytest.approx(0.25, **APPROX)

    def test_negative_exponent_backward(self):
        a = Value(4.0)
        c = a ** -1
        c.backward()
        assert a.grad == pytest.approx(-0.0625, **APPROX)  # −1·4^(−2) = −1/16

    def test_fractional_exponent_backward(self):
        a = Value(4.0)
        c = a ** 0.5
        c.backward()
        assert c.data == pytest.approx(2.0, **APPROX)
        assert a.grad == pytest.approx(0.25, **APPROX)  # 0.5·4^(−0.5) = 0.5/2


# --------------------------------------------------------------------------- #
# exp: c = exp(a)   →   ∂c/∂a = exp(a)
# --------------------------------------------------------------------------- #
class TestExp:
    def test_forward(self):
        c = Value(2.0).exp()
        assert c.data == pytest.approx(7.38905609893065, **APPROX)  # e²

    def test_backward(self):
        a = Value(2.0)
        c = a.exp()
        c.backward()
        assert a.grad == pytest.approx(7.38905609893065, **APPROX)  # e²

    def test_exp_zero(self):
        a = Value(0.0)
        c = a.exp()
        c.backward()
        assert c.data == pytest.approx(1.0, **APPROX)
        assert a.grad == pytest.approx(1.0, **APPROX)


# --------------------------------------------------------------------------- #
# tanh: c = tanh(a)   →   ∂c/∂a = 1 − tanh²(a)
# --------------------------------------------------------------------------- #
class TestTanh:
    def test_forward(self):
        c = Value(0.5).tanh()
        assert c.data == pytest.approx(0.46211715726000974, **APPROX)

    def test_backward(self):
        a = Value(0.5)
        c = a.tanh()
        c.backward()
        # 1 − tanh²(0.5) = 1 − 0.46211715726000974²
        assert a.grad == pytest.approx(0.7864477329659274, **APPROX)

    def test_tanh_zero(self):
        a = Value(0.0)
        c = a.tanh()
        c.backward()
        assert c.data == pytest.approx(0.0, **APPROX)
        assert a.grad == pytest.approx(1.0, **APPROX)  # 1 − 0²

    def test_tanh_saturation(self):
        a = Value(10.0)
        c = a.tanh()
        c.backward()
        assert c.data == pytest.approx(0.9999999958776927, **APPROX)
        # derivative is tiny in the saturated region
        assert a.grad == pytest.approx(8.244614768671e-09, rel=1e-6)


# --------------------------------------------------------------------------- #
# relu: c = max(0, a)   →   ∂c/∂a = 1 if a > 0 else 0
# --------------------------------------------------------------------------- #
class TestRelu:
    def test_forward_positive(self):
        assert Value(3.5).relu().data == pytest.approx(3.5, **APPROX)

    def test_forward_negative(self):
        assert Value(-2.0).relu().data == pytest.approx(0.0, **APPROX)

    def test_backward_positive(self):
        a = Value(3.5)
        c = a.relu()
        c.backward()
        assert a.grad == pytest.approx(1.0, **APPROX)

    def test_backward_negative(self):
        a = Value(-2.0)
        c = a.relu()
        c.backward()
        assert a.grad == pytest.approx(0.0, **APPROX)

    def test_backward_at_zero(self):
        # Convention: derivative at exactly 0 is 0.
        a = Value(0.0)
        c = a.relu()
        c.backward()
        assert a.grad == pytest.approx(0.0, **APPROX)


# --------------------------------------------------------------------------- #
# Derived ops — must be composed from the primitives.
# --------------------------------------------------------------------------- #
class TestNeg:
    def test_forward(self):
        assert (-Value(3.0)).data == pytest.approx(-3.0, **APPROX)

    def test_backward(self):
        a = Value(3.0)
        c = -a
        c.backward()
        assert a.grad == pytest.approx(-1.0, **APPROX)


class TestSub:
    def test_forward(self):
        c = Value(5.0) - Value(3.0)
        assert c.data == pytest.approx(2.0, **APPROX)

    def test_backward(self):
        a, b = Value(5.0), Value(3.0)
        c = a - b
        c.backward()
        assert a.grad == pytest.approx(1.0, **APPROX)
        assert b.grad == pytest.approx(-1.0, **APPROX)

    def test_sub_scalar(self):
        a = Value(5.0)
        c = a - 3.0
        c.backward()
        assert c.data == pytest.approx(2.0, **APPROX)
        assert a.grad == pytest.approx(1.0, **APPROX)


class TestDiv:
    def test_forward(self):
        c = Value(6.0) / Value(3.0)
        assert c.data == pytest.approx(2.0, **APPROX)

    def test_backward(self):
        a, b = Value(6.0), Value(3.0)
        c = a / b
        c.backward()
        assert a.grad == pytest.approx(1.0 / 3.0, **APPROX)   # ∂(a/b)/∂a = 1/b
        assert b.grad == pytest.approx(-6.0 / 9.0, **APPROX)  # ∂(a/b)/∂b = −a/b²

    def test_div_scalar(self):
        a = Value(6.0)
        c = a / 3.0
        c.backward()
        assert c.data == pytest.approx(2.0, **APPROX)
        assert a.grad == pytest.approx(1.0 / 3.0, **APPROX)


class TestReflected:
    def test_radd(self):
        a = Value(2.0)
        c = 3.0 + a
        c.backward()
        assert c.data == pytest.approx(5.0, **APPROX)
        assert a.grad == pytest.approx(1.0, **APPROX)

    def test_rmul(self):
        a = Value(2.0)
        c = 3.0 * a
        c.backward()
        assert c.data == pytest.approx(6.0, **APPROX)
        assert a.grad == pytest.approx(3.0, **APPROX)

    def test_rsub(self):
        a = Value(2.0)
        c = 5.0 - a
        c.backward()
        assert c.data == pytest.approx(3.0, **APPROX)
        assert a.grad == pytest.approx(-1.0, **APPROX)

    def test_rtruediv(self):
        a = Value(4.0)
        c = 8.0 / a
        c.backward()
        assert c.data == pytest.approx(2.0, **APPROX)
        assert a.grad == pytest.approx(-0.5, **APPROX)  # ∂(8/a)/∂a = −8/a² = −8/16


# --------------------------------------------------------------------------- #
# Chained compositions with hand-computed gradients.
# --------------------------------------------------------------------------- #
class TestComposite:
    def test_karpathy_neuron(self):
        """The classic worked example: a two-input neuron with tanh.

        x1=2, x2=0, w1=-3, w2=1, b=6.8813735870195432
        n = x1·w1 + x2·w2 + b ;  o = tanh(n)
        Hand-derived: o ≈ 0.7071, and
          x1.grad = w1·(1−o²) ≈ −1.5,  w1.grad = x1·(1−o²) ≈ 1.0
          x2.grad = w2·(1−o²) ≈ 0.5,   w2.grad = x2·(1−o²) = 0.0
        """
        x1, x2 = Value(2.0), Value(0.0)
        w1, w2 = Value(-3.0), Value(1.0)
        b = Value(6.8813735870195432)
        n = x1 * w1 + x2 * w2 + b
        o = n.tanh()
        o.backward()
        assert o.data == pytest.approx(0.7071067811865476, rel=1e-9)
        assert x1.grad == pytest.approx(-1.5, rel=1e-6)
        assert w1.grad == pytest.approx(1.0, rel=1e-6)
        assert x2.grad == pytest.approx(0.5, rel=1e-6)
        assert w2.grad == pytest.approx(0.0, abs=1e-9)

    def test_polynomial(self):
        """f(x) = 3x² − 4x + 5 at x = 2 → f = 9,  f' = 6x − 4 = 8."""
        x = Value(2.0)
        f = 3.0 * x ** 2 - 4.0 * x + 5.0
        f.backward()
        assert f.data == pytest.approx(9.0, **APPROX)
        assert x.grad == pytest.approx(8.0, **APPROX)

    def test_mixed_chain(self):
        """g = (a·b + b³) / a at a=2, b=3 →  g = (6+27)/2 = 16.5
        ∂g/∂a = b/a − (ab+b³)/a² = 3/2 − 33/4 = −6.75
        ∂g/∂b = (a + 3b²)/a = (2+27)/2 = 14.5
        """
        a, b = Value(2.0), Value(3.0)
        g = (a * b + b ** 3) / a
        g.backward()
        assert g.data == pytest.approx(16.5, **APPROX)
        assert a.grad == pytest.approx(-6.75, **APPROX)
        assert b.grad == pytest.approx(14.5, **APPROX)
