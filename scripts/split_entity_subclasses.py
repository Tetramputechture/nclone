#!/usr/bin/env python3
"""Split Entity subclasses from a monolithic entities.py into separate files.

This script:
- Parses a given `entities.py` using the Python AST
- Detects all classes that inherit from `Entity` (directly or indirectly)
- Writes each such class definition into its own file under a sibling package `entites/`
- Adds basic imports to each new file so it can be imported independently
- Removes those class definitions from the original `entities.py`

Notes:
- Folder name is intentionally `entites/` (as requested) to avoid clashing with `entities.py`.
- Base `Entity` and helper classes (e.g., `GridSegmentLinear`, `GridSegmentCircular`) remain in `entities.py`.
- Subclasses that inherit from other subclasses will import their base from the same `entites/` package.
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import sys
from typing import Dict, List, Set, Tuple


def camel_to_snake(name: str) -> str:
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def get_class_bases(node: ast.ClassDef) -> List[str]:
    base_names: List[str] = []
    for base in node.bases:
        # We expect simple names like Entity, EntityDoorBase, etc., not qualified names
        if isinstance(base, ast.Name):
            base_names.append(base.id)
        elif isinstance(base, ast.Attribute):
            # Handle possible qualified names, take the right-most attr
            base_names.append(base.attr)
        else:
            # Fallback textual representation
            base_names.append(ast.unparse(base) if hasattr(ast, "unparse") else "")
    return [b for b in base_names if b]


def build_inheritance_index(tree: ast.AST) -> Tuple[Dict[str, List[str]], Dict[str, ast.ClassDef]]:
    """Return (class_name -> base_names, class_name -> class_node)."""
    inheritance: Dict[str, List[str]] = {}
    class_nodes: Dict[str, ast.ClassDef] = {}
    for node in tree.body if isinstance(tree, ast.Module) else []:  # type: ignore[attr-defined]
        if isinstance(node, ast.ClassDef):
            class_nodes[node.name] = node
            inheritance[node.name] = get_class_bases(node)
    return inheritance, class_nodes


def compute_entity_subclasses(inheritance: Dict[str, List[str]]) -> Set[str]:
    """Compute all classes that inherit from Entity, directly or indirectly, excluding Entity itself."""
    subclasses: Set[str] = set()

    # Precompute parents map for reverse traversal
    parents_map: Dict[str, List[str]] = {}
    for cls, bases in inheritance.items():
        for base in bases:
            parents_map.setdefault(base, []).append(cls)

    # BFS/DFS from 'Entity' -> all descendants
    stack: List[str] = ["Entity"]
    seen: Set[str] = set()
    while stack:
        base = stack.pop()
        if base in seen:
            continue
        seen.add(base)
        for child in parents_map.get(base, []):
            if child != "Entity":
                subclasses.add(child)
            stack.append(child)

    return subclasses


def slice_source_by_node(source_lines: List[str], node: ast.ClassDef) -> Tuple[int, int, str]:
    """Return (start_idx, end_idx, text) for the class node within source_lines.

    start_idx and end_idx are 0-based, end exclusive.
    """
    # Python 3.8+ should have end_lineno
    if getattr(node, "lineno", None) is None or getattr(node, "end_lineno", None) is None:
        raise RuntimeError("AST nodes must include lineno/end_lineno; run with Python 3.8+.")
    start_idx = node.lineno - 1
    end_idx = node.end_lineno  # already exclusive once converted to 0-based
    text = "".join(source_lines[start_idx:end_idx])
    return start_idx, end_idx, text


def ensure_package_dir(package_dir: str) -> None:
    os.makedirs(package_dir, exist_ok=True)
    init_path = os.path.join(package_dir, "__init__.py")
    if not os.path.exists(init_path):
        with open(init_path, "w", encoding="utf-8") as f:
            f.write("# Auto-generated package for Entity subclasses\n")


def generate_module_source(
    class_name: str,
    class_code: str,
    class_bases: List[str],
    want_always_imports: List[str] | None = None,
) -> str:
    """Compose module content for a single Entity subclass file."""
    always_imports = want_always_imports or []
    lines: List[str] = []

    # Standard imports (keep minimal but safe)
    std_imports = ["import math", "import array", "import struct"]
    lines.extend(std_imports)
    lines.append("")

    # Project-level imports
    # Base Entity and helpers live in parent module `entities.py`
    lines.append("from ..entities import Entity, GridSegmentLinear, GridSegmentCircular")
    # Physics and ninja constants
    lines.append("from ..physics import *")
    lines.append("from ..ninja import NINJA_RADIUS")

    # Import base classes that are also Entity subclasses from the same package.
    for base in class_bases:
        if base == "Entity":
            continue
        # Import sibling base from `.entites`
        base_module = camel_to_snake(base)
        lines.append(f"from .{base_module} import {base}")

    # Any additional always-on imports
    for imp in always_imports:
        lines.append(imp)

    lines.append("")
    # Add the class code as-is
    # Trim leading/trailing newlines to avoid excessive vertical space, then ensure one newline before code
    class_code_stripped = class_code.lstrip("\n")
    lines.append(class_code_stripped)

    return "\n".join(lines)


def split_entities(entities_py_path: str, package_name: str = "entites") -> None:
    # Resolve paths
    entities_py_path = os.path.abspath(entities_py_path)
    parent_dir = os.path.dirname(entities_py_path)
    package_dir = os.path.join(parent_dir, package_name)

    with open(entities_py_path, "r", encoding="utf-8") as f:
        source = f.read()
    source_lines = list(source)
    # list(source) splits into chars; we need lines
    source_lines = source.splitlines(keepends=True)

    tree = ast.parse(source)
    inheritance, class_nodes = build_inheritance_index(tree)
    subclasses = compute_entity_subclasses(inheritance)

    if not subclasses:
        print("No subclasses of Entity found to split.")
        return

    ensure_package_dir(package_dir)

    # Prepare removals: collect ranges and class texts
    class_spans: List[Tuple[int, int, str, str, List[str]]] = []
    for cls_name in subclasses:
        node = class_nodes.get(cls_name)
        if not node:
            continue
        start_idx, end_idx, text = slice_source_by_node(source_lines, node)
        bases = inheritance.get(cls_name, [])
        class_spans.append((start_idx, end_idx, cls_name, text, bases))

    # Sort by start descending so removal doesn't shift positions
    class_spans.sort(key=lambda t: t[0], reverse=True)

    # Write each class to its own module
    for start_idx, end_idx, cls_name, class_text, bases in class_spans:
        module_name = camel_to_snake(cls_name)
        out_path = os.path.join(package_dir, f"{module_name}.py")
        module_src = generate_module_source(cls_name, class_text, bases)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(module_src)
        print(f"Wrote {out_path}")

    # Remove class definitions from the original file
    new_lines = source_lines[:]
    for start_idx, end_idx, cls_name, _class_text, _bases in class_spans:
        # Replace the range with a short comment placeholder to keep line numbers readable
        replacement = [f"\n# Moved class {cls_name} to ./{package_name}/{camel_to_snake(cls_name)}.py\n\n"]
        new_lines[start_idx:end_idx] = replacement

    # Write back modified entities.py
    with open(entities_py_path, "w", encoding="utf-8") as f:
        f.write("".join(new_lines))
    print(f"Updated {entities_py_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Split Entity subclasses to separate modules.")
    parser.add_argument(
        "entities_py",
        nargs="?",
        default=os.path.join(os.path.dirname(__file__), "..", "nclone", "entities.py"),
        help="Path to entities.py (default: nclone/nclone/entities.py relative to repo root)",
    )
    parser.add_argument(
        "--package-name",
        default="entites",
        help="Name of the target package directory to create (default: entites)",
    )
    args = parser.parse_args()

    # Normalize default path if left relative to this script location
    entities_py = os.path.abspath(args.entities_py)
    split_entities(entities_py, package_name=args.package_name)


if __name__ == "__main__":
    main()


