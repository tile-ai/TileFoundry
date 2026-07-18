from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

from tilefoundry.ir.target.storage import StorageKind, resolve_storage

from .dtype import DType
from .shape_dim import ShapeDim


def _canonicalize_static_dims(shape: tuple) -> tuple:
    """Fold any integer-valued ``Constant`` shape dim into a plain ``int`` so a
    static dim has a single canonical representation (``Slice`` / parser-sugar
    emit ``Constant`` while params / ``Reduce`` emit ``int``; mixing the two
    breaks ``==`` shape checks). Narrow on purpose: only a real integer
    ``Constant`` *in the shape tuple* is folded — ``DimVar`` and dynamic dim
    ``Call`` exprs pass through untouched. The ``Constant`` import is deferred to
    avoid the ``ir.core.expr`` ↔ ``ir.types.tensor_type`` cycle and fails closed
    (returns ``shape`` unchanged) so no non-``Constant`` object is ever folded."""
    try:
        from tilefoundry.ir.core.expr import Constant  # noqa: PLC0415 - cycle guard
    except ImportError:  # pragma: no cover - import-cycle guard, fail closed
        return shape
    out = []
    changed = False
    for d in shape:
        if isinstance(d, Constant) and isinstance(d.value, int) and not isinstance(d.value, bool):
            out.append(int(d.value))
            changed = True
        else:
            out.append(d)
    return tuple(out) if changed else shape


@dataclass(frozen=True)
class TensorType:
    shape: tuple[ShapeDim, ...]
    dtype: DType
    layout: "LayoutBase | None"
    # Memory space of the tensor. ``None`` marks a non-memory-resident
    # compile-time / shape scalar; a memory-resident tensor must carry a
    # concrete ``StorageKind``.
    storage: Optional[StorageKind]

    def __post_init__(self) -> None:
        # Normalise a canonical short-name string to ``StorageKind | None`` so
        # the IR instance never carries a raw string, even when constructed
        # directly (bypassing the parser / DSL surface).
        normalized = resolve_storage(self.storage)
        if normalized is not self.storage:
            object.__setattr__(self, "storage", normalized)
        # Canonicalize integer ``Constant`` dims to plain ``int``. Short-circuit
        # the common all-``int`` shape so the hot path skips the deferred import.
        if any(not isinstance(d, int) for d in self.shape):
            canon = _canonicalize_static_dims(self.shape)
            if canon is not self.shape:
                object.__setattr__(self, "shape", canon)

    @staticmethod
    def scalar(
        dtype: DType,
        layout: "LayoutBase | None" = None,
        storage: Optional[StorageKind] = StorageKind.RMEM,
    ) -> "TensorType":
        return TensorType(shape=(), dtype=dtype, layout=layout, storage=storage)


@dataclass(frozen=True)
class TupleType:
    fields: tuple[Union["TensorType", "TupleType"], ...]

@dataclass(frozen=True)
class UnitType:
    """Empty / void result type.

    Used as the ``Call.type`` of effect-ful TIR Ops that have no
    meaningful result value (the side effect is the operation itself —
    e.g. ``Copy``, ``Fill``, ``Mma``, ``ReLU`` writes to its ``dst``).
    Such Ops are placed in Stmt position via ``Evaluate(op, args)``.
    """

# ``CallableType`` is defined in ``callable_type.py`` and is the IR-level
# type of a callable Expr (``hir.Function``). It is part of this Union;
# the ``"CallableType"`` forward-ref keeps this module import-cycle-free.
Type = Union[TensorType, TupleType, UnitType, "CallableType"]

__all__ = ["DType", "TensorType", "TupleType", "UnitType", "Type"]
