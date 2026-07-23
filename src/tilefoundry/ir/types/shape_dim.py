"""``ShapeDim`` alias for ``TensorType.shape`` entries.

A ``ShapeDim`` is one of:

- a static ``int`` (compile-time dim);
- a ``DimVar`` ``Op`` instance (bounded named symbolic dim placed
  directly in ``TensorType.shape``); or
- a dynamic dim ``Expr`` (a ``Call`` / ``Var`` / ``Constant`` tree
  built from the ``DimConst`` / ``DimAdd`` / ``DimMul`` / ``DimMod``
  / ``DimMin`` / ``DimMax`` ops in ``tilefoundry.ir.types.dim``).

This module imports nothing from ``tilefoundry.ir.core``. The alias
exists so ``tilefoundry.ir.types.tensor_type`` can spell its shape
type without needing to import ``Expr`` (which would close the
``ir.core.expr`` ↔ ``ir.types.tensor_type`` cycle).

The body of the alias is kept as a string forward-ref so the alias
declaration itself never evaluates ``Expr`` / ``DimVar`` at runtime.
``shape`` is only consumed in annotations under
``from __future__ import annotations``, so the alias body is never
needed at runtime either.
"""

from __future__ import annotations

type ShapeDim = "int | DimVar | Expr"
