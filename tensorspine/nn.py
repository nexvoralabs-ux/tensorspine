"""tensorspine.nn — Neuron / Layer / MLP built on the Value engine.

Forward passes and weight initialization are YOUR WORK (stubs below).
Parameter traversal and zero_grad are plumbing and are implemented for you.

Stdlib only. Do not import numpy, torch, or anything else here.
"""

import random  # noqa: F401  (you will need it for weight initialization)

from tensorspine.engine import Value


class Module:
    """Base class providing the shared plumbing: parameters() and zero_grad()."""

    def parameters(self):
        """Return a flat list of every trainable Value in this module.

        Subclasses override this. The base returns an empty list.
        """
        return []

    def zero_grad(self):
        """Reset the .grad of every parameter to 0.0.

        Call this before each backward pass during training — gradients
        ACCUMULATE across backward calls, they do not reset themselves.
        (Experiment 03 demonstrates exactly what goes wrong if you skip this.)
        """
        for p in self.parameters():
            p.grad = 0.0


class Neuron(Module):
    """A single neuron: an affine map of its inputs followed by a nonlinearity.

    Holds ``nin`` weights and one bias, all of them ``Value`` instances so
    gradients flow into them.
    """

    def __init__(self, nin, nonlin="tanh", init="uniform"):
        """Create a neuron with ``nin`` inputs.

        Args:
            nin:    number of inputs this neuron receives.
            nonlin: 'tanh', 'relu', or 'linear' (no nonlinearity — used for
                    output layers doing regression / raw scores).
            init:   weight initialization scheme, 'uniform' or 'xavier'.
                    Two schemes are to be tried in this project:
                      - naive:  each weight drawn from uniform(-1, 1)
                      - Xavier: scaled by 1/sqrt(nin)
                    Experiment 04 compares how they converge. Both must be
                    available here, selected by this argument. The bias may
                    start at 0.

        YOUR WORK: create self.w (list of ``nin`` Value weights) and self.b
        (a Value bias) according to the chosen scheme.
        """
        self.nin = nin
        self.nonlin = nonlin
        self.init = init
        raise NotImplementedError("YOUR WORK: see docstring for the gradient rule")

    def __call__(self, x):
        """Forward pass:  activation( Σᵢ wᵢ·xᵢ + b ).

        Args:
            x: list of ``nin`` inputs (Values or plain numbers).

        Returns:
            a single Value — the weighted sum plus bias, passed through this
            neuron's nonlinearity ('tanh' → .tanh(), 'relu' → .relu(),
            'linear' → no nonlinearity).

        The whole computation must be built from Value operations so the
        backward pass can flow through it.
        """
        raise NotImplementedError("YOUR WORK: see docstring for the gradient rule")

    def parameters(self):
        """All trainable Values of this neuron: the weights, then the bias."""
        return self.w + [self.b]

    def __repr__(self):
        return f"Neuron(nin={self.nin}, nonlin={self.nonlin!r}, init={self.init!r})"


class Layer(Module):
    """A layer: ``nout`` independent Neurons, each seeing the same input vector."""

    def __init__(self, nin, nout, **kwargs):
        """Create a layer of ``nout`` neurons with ``nin`` inputs each.

        Extra keyword arguments (nonlin=, init=) are forwarded to each Neuron.
        This is plumbing — implemented for you.
        """
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        """Forward pass: apply every neuron to the same input list ``x``.

        Returns:
            a list of Values, one per neuron — EXCEPT when the layer has exactly
            one neuron, in which case return the single Value bare (not wrapped
            in a list). This makes 1-output networks pleasant to use.
        """
        raise NotImplementedError("YOUR WORK: see docstring for the gradient rule")

    def parameters(self):
        """All parameters of all neurons, flattened into one list."""
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer(of={len(self.neurons)} neurons)"


class MLP(Module):
    """A multi-layer perceptron: a chain of Layers.

    By convention the hidden layers use the given nonlinearity and the final
    layer is linear (raw output), which suits both regression and pre-sigmoid
    classification scores.
    """

    def __init__(self, nin, nouts, **kwargs):
        """Create an MLP.

        Args:
            nin:   number of inputs to the network.
            nouts: list of layer sizes, e.g. [16, 16, 1] for two hidden layers
                   of 16 and a single output.
            kwargs: forwarded to Layers/Neurons (nonlin=, init=). The final
                   layer is forced to nonlin='linear' regardless.

        This is plumbing — implemented for you.
        """
        sizes = [nin] + list(nouts)
        self.layers = []
        for i in range(len(nouts)):
            layer_kwargs = dict(kwargs)
            if i == len(nouts) - 1:
                layer_kwargs["nonlin"] = "linear"
            self.layers.append(Layer(sizes[i], sizes[i + 1], **layer_kwargs))

    def __call__(self, x):
        """Forward pass: feed ``x`` through each layer in sequence.

        Args:
            x: list of ``nin`` inputs (Values or plain numbers).

        Returns:
            whatever the final layer returns (a list of Values, or a bare Value
            if the final layer has one neuron).
        """
        raise NotImplementedError("YOUR WORK: see docstring for the gradient rule")

    def parameters(self):
        """All parameters of all layers, flattened into one list."""
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP(layers={self.layers})"
