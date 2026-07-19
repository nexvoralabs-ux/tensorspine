"""tensorspine.viz — graphviz rendering of the computation DAG.

Fully implemented — this is a debugging tool, not part of the learning work.
It walks a Value's graph using only the public-ish attributes (.data, .grad,
._prev, ._op) and never touches gradient logic, so it works from day one even
while the engine is unimplemented.

Usage:
    from tensorspine import Value
    from tensorspine.viz import draw_dot

    # ... build an expression ending in some Value `loss` ...
    draw_dot(loss).render("graph", format="png", cleanup=True)
    # or in a notebook: draw_dot(loss)  # renders inline

Degrades gracefully with a clear message if graphviz is not installed.
"""


def _trace(root):
    """Collect every node and edge reachable from ``root``.

    Returns:
        (nodes, edges): a set of Values and a set of (parent, child) pairs.

    Pure graph walking — no gradient logic, no ordering requirements.
    """
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for parent in v._prev:
                edges.add((parent, v))
                build(parent)

    build(root)
    return nodes, edges


def draw_dot(root, rankdir="LR"):
    """Render the computation graph rooted at ``root`` as a graphviz Digraph.

    Each Value is drawn as a record box showing its ``data`` and ``grad``.
    Each operation is drawn as a small oval labelled with the op symbol, with
    edges  parents → op → result.

    Args:
        root:    the output Value whose graph to draw (typically the loss).
        rankdir: 'LR' (left-to-right, default) or 'TB' (top-to-bottom).

    Returns:
        a ``graphviz.Digraph``. Call ``.render(filename, format='png')`` to
        write an image, or display it directly in a notebook.

    Raises:
        RuntimeError with an actionable message if the graphviz Python package
        is not installed.
    """
    try:
        from graphviz import Digraph
    except ImportError as err:
        raise RuntimeError(
            "tensorspine.viz needs the 'graphviz' Python package.\n"
            "  Install it with:            pip install graphviz\n"
            "  You ALSO need the system Graphviz binaries (the 'dot' tool):\n"
            "    macOS:          brew install graphviz\n"
            "    Debian/Ubuntu:  apt-get install graphviz\n"
            "    Windows:        choco install graphviz  (or graphviz.org downloads)\n"
        ) from err

    if rankdir not in ("LR", "TB"):
        raise ValueError(f"rankdir must be 'LR' or 'TB', got {rankdir!r}")

    nodes, edges = _trace(root)

    dot = Digraph(format="svg", graph_attr={"rankdir": rankdir})

    for n in nodes:
        uid = str(id(n))
        # Record box: | op-label? | data | grad |
        dot.node(
            name=uid,
            label=f"{{ data {n.data:.4f} | grad {n.grad:.4f} }}",
            shape="record",
        )
        if n._op:
            # A distinct oval node for the operation that produced n.
            dot.node(name=uid + n._op, label=n._op)
            dot.edge(uid + n._op, uid)

    for parent, child in edges:
        # Edge from parent's value box into the op node of the child.
        dot.edge(str(id(parent)), str(id(child)) + child._op)

    return dot


def save_graph(root, filename="graph", fmt="png", rankdir="LR"):
    """Convenience wrapper: render the graph of ``root`` straight to a file.

    Args:
        root:     the output Value whose graph to draw.
        filename: output path without extension (graphviz appends it).
        fmt:      'png', 'svg', 'pdf', ...
        rankdir:  'LR' or 'TB'.

    Returns:
        the path of the written file (str), or None if rendering failed because
        the system 'dot' binary is missing (a clear message is printed instead
        of raising, so a missing binary never crashes a training script).
    """
    dot = draw_dot(root, rankdir=rankdir)
    dot.format = fmt
    try:
        return dot.render(filename, cleanup=True)
    except Exception as err:  # graphviz.ExecutableNotFound and friends
        print(
            f"[tensorspine.viz] Could not render graph to {filename}.{fmt}: {err}\n"
            "  The graphviz Python package is present but the system 'dot' binary\n"
            "  appears to be missing or broken. Install Graphviz:\n"
            "    macOS:          brew install graphviz\n"
            "    Debian/Ubuntu:  apt-get install graphviz\n"
            "    Windows:        choco install graphviz  (or graphviz.org downloads)"
        )
        return None
