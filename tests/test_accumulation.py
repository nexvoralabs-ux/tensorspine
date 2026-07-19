"""Gradient ACCUMULATION tests — the highest-value file in the suite.

Every test here catches the same family of bug: writing ``grad = ...`` instead
of ``grad += ...`` inside a backward closure. With ``=``, the last path to reach
a shared node OVERWRITES the contributions of every other path; with ``+=`` the
paths correctly SUM (the multivariate chain rule).

If test_ops.py passes but this file fails, you have the = vs += bug.
"""

import pytest

from tensorspine import Value

APPROX = dict(rel=1e-9, abs=1e-12)


class TestSelfUse:
    def test_a_plus_a(self):
        """c = a + a  →  a.grad must be 2.

        Bug caught: with ``=`` instead of ``+=``, the second contribution
        overwrites the first and a.grad comes out 1.0 instead of 2.0.
        """
        a = Value(3.0)
        c = a + a
        c.backward()
        assert c.data == pytest.approx(6.0, **APPROX)
        assert a.grad == pytest.approx(2.0, **APPROX)

    def test_a_times_a(self):
        """c = a * a  →  a.grad must be 2·a.data.

        Bug caught: with ``=``, one of the two ∂c/∂a = a contributions is lost
        and a.grad comes out a.data (3.0) instead of 2·a.data (6.0).
        """
        a = Value(3.0)
        c = a * a
        c.backward()
        assert c.data == pytest.approx(9.0, **APPROX)
        assert a.grad == pytest.approx(6.0, **APPROX)

    def test_a_cubed_via_repeated_mul(self):
        """c = a * a * a  →  a.grad must be 3·a².

        Bug caught: a appears on three multiplicative paths at two different
        graph depths. Overwriting OR visiting nodes in the wrong order (a
        broken topological sort) both give a wrong answer here.
        """
        a = Value(2.0)
        c = a * a * a
        c.backward()
        assert c.data == pytest.approx(8.0, **APPROX)
        assert a.grad == pytest.approx(12.0, **APPROX)  # 3·2²


class TestDiamond:
    def test_two_path_diamond(self):
        """d = (a*b) + (a*c)  →  a.grad must be b.data + c.data.

        Bug caught: ``a`` fans out into two multiplication branches that later
        merge in the add. With ``=``, whichever branch runs backward last wins,
        so a.grad is b.data OR c.data instead of their sum.
        """
        a, b, c = Value(2.0), Value(3.0), Value(5.0)
        d = (a * b) + (a * c)
        d.backward()
        assert d.data == pytest.approx(16.0, **APPROX)
        assert a.grad == pytest.approx(8.0, **APPROX)   # b + c = 3 + 5
        assert b.grad == pytest.approx(2.0, **APPROX)   # a
        assert c.grad == pytest.approx(2.0, **APPROX)   # a

    def test_three_path_diamond(self):
        """f = a*b + a*c + a*d  →  a.grad must be b + c + d.

        Bug caught: three shared paths. An overwrite bug leaves a.grad equal to
        just one of the three terms; a partially-working accumulate (e.g. one
        op uses += and another uses =) leaves it equal to a partial sum.
        """
        a = Value(2.0)
        b, c, d = Value(3.0), Value(5.0), Value(7.0)
        f = a * b + a * c + a * d
        f.backward()
        assert f.data == pytest.approx(30.0, **APPROX)
        assert a.grad == pytest.approx(15.0, **APPROX)  # 3 + 5 + 7

    def test_deep_diamond_reconverging(self):
        """A diamond whose branches have different depths before re-merging.

        e = a * a       (depth-1 branch, itself a self-use)
        f = (a + 1) * a (depth-2 branch reusing a twice more)
        g = e + f

        At a = 3:  g = 9 + 12 = 21
        ∂g/∂a = 2a + (2a + 1) = 4a + 1 = 13

        Bug caught: correct answers here require BOTH accumulation and a
        correct reverse-topological visiting order — a node's own closure must
        not run until every consumer has deposited its contribution.
        """
        a = Value(3.0)
        e = a * a
        f = (a + 1.0) * a
        g = e + f
        g.backward()
        assert g.data == pytest.approx(21.0, **APPROX)
        assert a.grad == pytest.approx(13.0, **APPROX)


class TestBranchingActivations:
    def test_node_feeds_tanh_and_relu(self):
        """One node consumed by BOTH a tanh branch and a relu branch.

        s = tanh(a) + relu(a) at a = 0.5:
          ∂s/∂a = (1 − tanh²(0.5)) + 1 = 0.7864477329659274 + 1

        Bug caught: shared use across two DIFFERENT op types. Catches the case
        where one activation's closure accumulates correctly and the other
        overwrites (each op's closure is written separately, so each can carry
        its own copy of the bug).
        """
        a = Value(0.5)
        s = a.tanh() + a.relu()
        s.backward()
        assert s.data == pytest.approx(0.46211715726000974 + 0.5, **APPROX)
        assert a.grad == pytest.approx(1.7864477329659274, rel=1e-9)

    def test_node_feeds_tanh_and_relu_negative(self):
        """Same shape at a = −0.5, where the relu branch is dead.

        s = tanh(a) + relu(a):
          ∂s/∂a = (1 − tanh²(−0.5)) + 0 = 0.7864477329659274

        Bug caught: an overwrite bug where the dead relu branch runs LAST would
        zero out the tanh contribution entirely — a.grad would be 0.0.
        """
        a = Value(-0.5)
        s = a.tanh() + a.relu()
        s.backward()
        assert a.grad == pytest.approx(0.7864477329659274, rel=1e-9)


class TestRepeatedBackwardSemantics:
    def test_grads_accumulate_across_backward_calls(self):
        """Calling backward() twice without zeroing doubles the grads.

        This is the DEFINED semantics of accumulation (and what zero_grad()
        exists to reset). c = a * b, backward twice → a.grad == 2·b.data.

        Bug caught: an implementation that resets grads inside backward() —
        which would silently break gradient accumulation workflows and make
        experiment 03 meaningless.
        """
        a, b = Value(2.0), Value(3.0)
        c = a * b
        c.backward()
        c.backward()
        assert a.grad == pytest.approx(6.0, **APPROX)   # 2 × 3.0
        assert b.grad == pytest.approx(4.0, **APPROX)   # 2 × 2.0
