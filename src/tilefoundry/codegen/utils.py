"""Unique function paths and generated symbols."""

import re

from tilefoundry.ir.core.module import Module, function_paths
from tilefoundry.ir.hir.specialize import display_name


def function_path(root: Module, fn) -> tuple[str, ...]:
    """Use the selector traversal for prototypes and their specialization labels."""
    matches = []
    for path, function in function_paths(root):
        if function is fn:
            matches.append((root.name, *path))
        for index, variant in enumerate(function.variants or ()):
            if variant is fn:
                matches.append((root.name, *path, display_name(variant) or f"v{index}"))
    if len(matches) != 1:
        raise ValueError(f"codegen: function {fn.name!r} has no unique path in {root.name!r}")
    return matches[0]


def mangle(root: Module, fn) -> str:
    """Encode each path segment with its length; reject non-C identifiers."""
    path = function_path(root, fn)
    for segment in path:
        if re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", segment) is None:
            raise ValueError(f"codegen: invalid path segment {segment!r} in {'.'.join(path)!r}")
    return "tilefoundry_" + "".join(f"{len(segment)}{segment}" for segment in path)


def export_name(root: Module, fn) -> str:
    """The dotted TVM FFI registration name corresponding to a selector."""
    return ".".join(function_path(root, fn))
