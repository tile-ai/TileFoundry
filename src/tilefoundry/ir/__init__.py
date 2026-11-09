"""tilefoundry.ir — the IR tree, one subpackage per layer.

Bottom to top: `ir.expr` (expressions, operations, dimension nodes), `ir.types`
(type, layout and mesh definitions), `ir.dim` (dimension behaviour), `ir.utils`
(helpers over types), `ir.core` (structure nodes and visitors), `ir.clause` and
`ir.pattern` (where-clauses and declaration patterns), `ir.hir` and `ir.tir`.
Each subpackage declares its own exports. This package imports none of them,
so importing a low layer such as `ir.expr` does not load the layers above it.
"""

from __future__ import annotations
