from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

from tilefoundry.ir.types.storage import StorageKind, resolve_storage

from .dtype import DType
from .shape_dim import ShapeDim


def _canonicalize_static_dims(shape: tuple) -> tuple:
    """Replace integer ``Constant`` shape entries with canonical plain integers.

    Symbolic and dynamic expressions pass through. The deferred import avoids
    an expression/type cycle and fails closed by returning the original shape.

    See [types §4](docs/spec/types.md#4-dim--symbolic-shape-dimensions).
    """
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

    storage: Optional[StorageKind]

    def __post_init__(self) -> None:

        normalized = resolve_storage(self.storage)
        if normalized is not self.storage:
            object.__setattr__(self, "storage", normalized)

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

    @staticmethod
    def meta_scalar(dtype: DType = DType.i64) -> "TensorType":
        """Canonical rank-0 shape/meta scalar.

        Canonical rank-0 shape/meta scalar (``layout=EMPTY_LAYOUT``,
        ``storage=None`` — a non-memory-resident compile-time value).
        Every shape-element / dim-arithmetic type must use this single
        form so structural type equality holds across construction sites.
        """
        from .shard.layout import EMPTY_LAYOUT  # noqa: PLC0415 - cycle guard

        return TensorType(shape=(), dtype=dtype, layout=EMPTY_LAYOUT, storage=None)


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


Type = Union[TensorType, TupleType, UnitType, "CallableType"]

__all__ = ["DType", "TensorType", "TupleType", "UnitType", "Type"]
