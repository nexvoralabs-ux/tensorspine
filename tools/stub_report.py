#!/usr/bin/env python3
"""Report remaining NotImplementedError stubs in the hand-written engine.

Non-engine tooling. Used by `make stubs` and `make progress`. Reads the two
files the human implements by hand and lists every method still raising
NotImplementedError, so the count in the README and the count the tooling
reports can never silently drift apart.

stdlib only — runs before the venv exists.
"""

from __future__ import annotations

import pathlib
import re

FILES = ["tensorspine/engine.py", "tensorspine/nn.py"]


def find_stubs(path: pathlib.Path):
    """Return [(qualified_name, line_number), ...] for each stub in `path`."""
    cls = None
    func = None
    out = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        top_class = re.match(r"class\s+(\w+)", line)
        if top_class and not line[:1].isspace():
            cls = top_class.group(1)
        a_def = re.match(r"\s*def\s+(\w+)", line)
        if a_def:
            func = a_def.group(1)
        if "raise NotImplementedError" in line:
            name = f"{cls}.{func}" if cls else str(func)
            out.append((name, i))
    return out


def main() -> int:
    root = pathlib.Path(__file__).resolve().parent.parent
    total = 0
    print("Remaining stubs (raise NotImplementedError):")
    for rel in FILES:
        path = root / rel
        stubs = find_stubs(path)
        total += len(stubs)
        print(f"\n  {rel}  ({len(stubs)} remaining):")
        for name, ln in stubs:
            print(f"    - {name}  (line {ln})")
    print(f"\nTotal stub sites remaining: {total}")
    return total


if __name__ == "__main__":
    main()
