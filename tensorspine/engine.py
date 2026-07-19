"""tensorspine.engine — the scalar autograd core.

This module is INTENTIONALLY UNIMPLEMENTED. Every method that computes a
derivative, distributes a gradient, or walks the graph in reverse is a stub that
raises ``NotImplementedError``. The docstrings state the math you must implement;
they never state the code.

Stdlib only. Do not import numpy, torch, or anything else here.
"""

import math  # noqa: F401  (you will need it for exp / tanh; kept for convenience)


class Value:
    """A scalar node in a computation graph.

    A ``Value`` wraps a single ``float`` and remembers how it was produced, so
    that a reverse pass can compute the derivative of some final scalar loss with
    respect to every node that fed into it.

    Attributes:
        data:      float, the forward value.
        grad:      float, dL/d(self), ACCUMULATED across all paths. Starts at 0.
        _prev:     set of parent Values (the immediate inputs to the op that
                   produced this node) — these are the graph edges.
        _backward: a zero-argument closure that, given this node's ``grad``
                   already filled in, pushes the correct contribution onto each
                   parent's ``grad``. For a leaf it does nothing.
        _op:       str label naming the operation that produced this node
                   ('' for a leaf). Debugging / visualization only.
    """

    # ------------------------------------------------------------------ #
    # Plumbing — implemented for you. No math lives here.                 #
    # ------------------------------------------------------------------ #
    def __init__(self, data, _children=(), _op=""):
        """Create a node.

        Args:
            data:      the scalar forward value (int/float; stored as-is).
            _children: iterable of parent Values this node was built from.
            _op:       label of the producing operation ('' for a leaf).
        """
        self.data = data
        self.grad = 0.0
        # The default backward for a leaf is a no-op: leaves have no parents to
        # distribute to. Operations REPLACE this with their own closure.
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad}, op={self._op!r})"

    # ------------------------------------------------------------------ #
    # Forward operations — YOUR WORK. Each builds a new Value AND wires   #
    # up its ._backward closure. No code below the docstrings.           #
    # ------------------------------------------------------------------ #
    def __add__(self, other):
        """Addition:  c = a + b.

        Local derivatives:
            ∂c/∂a = 1
            ∂c/∂b = 1

        The backward closure must route c.grad to both parents scaled by the
        local derivative above (the chain rule).

        CRITICAL: accumulate with += never =. A node used by multiple
        consumers receives the SUM of gradients over all paths.
        """
        raise NotImplementedError("YOUR WORK: see docstring for the gradient rule")

    def __mul__(self, other):
        """Multiplication:  c = a * b.

        Local derivatives:
            ∂c/∂a = b
            ∂c/∂b = a

        The backward closure must route c.grad to each parent scaled by the
        OTHER operand (the chain rule).

        CRITICAL: accumulate with += never =. A node used by multiple
        consumers receives the SUM of gradients over all paths.
        """
        raise NotImplementedError("YOUR WORK: see docstring for the gradient rule")

    def __pow__(self, k):
        """Power by a constant exponent:  c = a ** k,  k a real constant.

        Only ``k`` constant (int/float) need be supported — not Value ** Value.

        Local derivative:
            ∂c/∂a = k · a^(k−1)

        The backward closure must route c.grad to the base scaled by the local
        derivative above.

        CRITICAL: accumulate with += never =. A node used by multiple
        consumers receives the SUM of gradients over all paths.
        """
        raise NotImplementedError("YOUR WORK: see docstring for the gradient rule")

    def exp(self):
        """Exponential:  c = exp(a).

        Local derivative:
            ∂c/∂a = exp(a)          (i.e. the forward output itself)

        The backward closure must route c.grad to the parent scaled by the local
        derivative above.

        CRITICAL: accumulate with += never =. A node used by multiple
        consumers receives the SUM of gradients over all paths.
        """
        raise NotImplementedError("YOUR WORK: see docstring for the gradient rule")

    def tanh(self):
        """Hyperbolic tangent:  c = tanh(a).

        Local derivative:
            ∂c/∂a = 1 − tanh²(a)    (i.e. 1 minus the square of the forward output)

        The backward closure must route c.grad to the parent scaled by the local
        derivative above.

        CRITICAL: accumulate with += never =. A node used by multiple
        consumers receives the SUM of gradients over all paths.
        """
        raise NotImplementedError("YOUR WORK: see docstring for the gradient rule")

    def relu(self):
        """Rectified linear unit:  c = max(0, a).

        Local derivative:
            ∂c/∂a = 1 if a > 0 else 0

        (At exactly a = 0 the derivative is undefined; the standard convention is
        to take it as 0.)

        The backward closure must route c.grad to the parent scaled by the local
        derivative above.

        CRITICAL: accumulate with += never =. A node used by multiple
        consumers receives the SUM of gradients over all paths.
        """
        raise NotImplementedError("YOUR WORK: see docstring for the gradient rule")

    # ------------------------------------------------------------------ #
    # Derived operations — YOUR WORK, but COMPOSE from the primitives     #
    # above. Do NOT wire new _backward closures for these; build them out #
    # of __add__ / __mul__ / __pow__ / __neg__ so the chain rule flows    #
    # through the primitives you already implemented.                     #
    # ------------------------------------------------------------------ #
    def __neg__(self):
        """Negation:  -a.

        Must be COMPOSED from the primitives (a negation is a multiplication by
        the constant −1). Do not attach a new backward closure of its own.

        CRITICAL: accumulate with += never =. A node used by multiple
        consumers receives the SUM of gradients over all paths.
        """
        raise NotImplementedError("YOUR WORK: see docstring for the gradient rule")

    def __sub__(self, other):
        """Subtraction:  a - b.

        Must be COMPOSED from the primitives (a subtraction is an addition with a
        negation). Do not attach a new backward closure of its own.

        CRITICAL: accumulate with += never =. A node used by multiple
        consumers receives the SUM of gradients over all paths.
        """
        raise NotImplementedError("YOUR WORK: see docstring for the gradient rule")

    def __truediv__(self, other):
        """True division:  a / b.

        Must be COMPOSED from the primitives (a division is a multiplication by a
        reciprocal, and a reciprocal is a power of −1). Do not attach a new
        backward closure of its own.

        CRITICAL: accumulate with += never =. A node used by multiple
        consumers receives the SUM of gradients over all paths.
        """
        raise NotImplementedError("YOUR WORK: see docstring for the gradient rule")

    # ------------------------------------------------------------------ #
    # Reflected operators — YOUR WORK, COMPOSE from the primitives so that #
    # (scalar OP Value) works, e.g. 2 + x, 3 * x, 1 - x, 2 / x.           #
    # ------------------------------------------------------------------ #
    def __radd__(self, other):
        """Reflected add:  other + self, when ``other`` is a plain number.

        Must be COMPOSED from the primitives. Do not attach a new backward
        closure of its own.

        CRITICAL: accumulate with += never =. A node used by multiple
        consumers receives the SUM of gradients over all paths.
        """
        raise NotImplementedError("YOUR WORK: see docstring for the gradient rule")

    def __rmul__(self, other):
        """Reflected multiply:  other * self, when ``other`` is a plain number.

        Must be COMPOSED from the primitives. Do not attach a new backward
        closure of its own.

        CRITICAL: accumulate with += never =. A node used by multiple
        consumers receives the SUM of gradients over all paths.
        """
        raise NotImplementedError("YOUR WORK: see docstring for the gradient rule")

    def __rsub__(self, other):
        """Reflected subtract:  other - self, when ``other`` is a plain number.

        Must be COMPOSED from the primitives. Do not attach a new backward
        closure of its own.

        CRITICAL: accumulate with += never =. A node used by multiple
        consumers receives the SUM of gradients over all paths.
        """
        raise NotImplementedError("YOUR WORK: see docstring for the gradient rule")

    def __rtruediv__(self, other):
        """Reflected divide:  other / self, when ``other`` is a plain number.

        Must be COMPOSED from the primitives. Do not attach a new backward
        closure of its own.

        CRITICAL: accumulate with += never =. A node used by multiple
        consumers receives the SUM of gradients over all paths.
        """
        raise NotImplementedError("YOUR WORK: see docstring for the gradient rule")

    # ------------------------------------------------------------------ #
    # Reverse pass — YOUR WORK.                                           #
    # ------------------------------------------------------------------ #
    def _build_topo(self):
        """Return a list of every Value in this node's graph in TOPOLOGICAL order.

        Topological order means: a node appears only after all of its parents
        (``_prev``) appear. Build it with a depth-first traversal from ``self``,
        marking nodes visited so each is emitted exactly once, appending a node
        only after recursing into its parents.

        The reverse of this list is the order in which ``backward`` must invoke
        the per-node ``_backward`` closures.
        """
        raise NotImplementedError("YOUR WORK: see docstring for the gradient rule")

    def backward(self):
        """Run reverse-mode autodiff with ``self`` as the scalar output (the loss).

        Steps, in words (not code):
          1. Build the topological order of the graph reachable from ``self``.
          2. Seed the output's own gradient:  ∂L/∂L = 1, i.e. self.grad = 1.
          3. Walk the nodes in REVERSE topological order and call each node's
             ``_backward`` closure. Because every consumer of a node is visited
             before the node itself, each node's ``grad`` is fully accumulated by
             the time its own closure runs and pushes to its parents.

        This method does not return anything; it fills in ``.grad`` on every node
        in the graph.

        CRITICAL: accumulate with += never =. A node used by multiple
        consumers receives the SUM of gradients over all paths.
        """
        raise NotImplementedError("YOUR WORK: see docstring for the gradient rule")
