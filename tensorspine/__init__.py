"""tensorspine — a scalar reverse-mode autograd engine (learning project).

The engine (`engine.py`, `nn.py`) is stdlib-only. See README.md: the gradient
logic is intentionally left as stubs for the human to implement.
"""

from tensorspine.engine import Value
from tensorspine.nn import MLP, Layer, Neuron

__all__ = ["Value", "Neuron", "Layer", "MLP"]
