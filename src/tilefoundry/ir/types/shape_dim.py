"""Type alias for static, symbolic, or expression-valued shape entries.

The string forward reference avoids an import cycle between core expressions
and tensor types; annotations never require evaluating it at runtime.

See [types §4](docs/spec/types.md#4-dim--symbolic-shape-dimensions).
"""

from __future__ import annotations

type ShapeDim = "int | DimVar | Expr"
