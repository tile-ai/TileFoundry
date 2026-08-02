"""Shared machinery that names no layer.

What belongs here imports nothing from `ir/`, `parser/`, `passes/`, `codegen/`,
`runtime/` or `cli/`, and would read the same if any of them were deleted. A
module here is a leaf: it is depended on, it does not depend. That is the whole
bar, and it is what lets both the package and a standalone script -- a
pre-commit hook, say, running under an interpreter with nothing installed --
reach the same one implementation.

A helper that needs to know about a layer belongs in that layer instead.
"""

from __future__ import annotations

__all__: list[str] = []
