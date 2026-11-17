"""Value types: tensors, tuples of values, and the ``Type`` union."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

from tilefoundry.ir import expr

from .layout import Layout

__all__ = ["ShapeDim", "TensorType", "TupleType", "Type"]

type ShapeDim = "int | DimVar | Expr"
"""A shape entry: static, symbolic, or expression-valued."""


@dataclass(frozen=True)
class TensorType:
    """A ranked value: its shape, element type, layout and storage level.

    See [types §2](docs/spec/types.md#2-tensortype).
    """

    shape: tuple[ShapeDim, ...]
    dtype: expr.DType
    layout: "LayoutBase | None"

    storage: expr.StorageKind

    def __post_init__(self) -> None:
        """Normalize the storage and replace integer ``Constant`` extents by plain integers.

        Symbolic and dynamic expressions pass through.

        See [types §4](docs/spec/types.md#4-dim--symbolic-shape-dimensions).
        """
        normalized = expr.resolve_storage(self.storage)
        if normalized is None:
            raise TypeError("TensorType.storage must be a StorageKind, not None")
        if normalized is not self.storage:
            object.__setattr__(self, "storage", normalized)

        if any(not isinstance(d, int) for d in self.shape):
            out = []
            changed = False
            for d in self.shape:
                if isinstance(d, expr.Constant) and isinstance(d.value, int) and not isinstance(d.value, bool):
                    out.append(int(d.value))
                    changed = True
                else:
                    out.append(d)
            if changed:
                object.__setattr__(self, "shape", tuple(out))

    @classmethod
    def from_scalar(
        cls,
        dtype: expr.DType,
        layout: "LayoutBase | None" = None,
        storage: expr.StorageKind = expr.StorageKind.UMAT,
    ) -> TensorType:
        """A rank-0 value of *dtype*."""
        return cls(
            shape=(),
            dtype=dtype,
            layout=Layout(()) if layout is None else layout,
            storage=storage,
        )

    @classmethod
    def from_umat_scalar(cls, dtype: expr.DType = expr.DType.i64) -> TensorType:
        """Canonical rank-0 value whose residency is not decided.

        Shape elements and dimension arithmetic results use this single form
        so structural type equality holds across construction sites.
        """
        return cls.from_scalar(dtype)

    @classmethod
    def from_umat_tensor(cls, shape: tuple, dtype: expr.DType = expr.DType.i64) -> TensorType:
        """Canonical ranked value whose residency is not decided."""
        shape = tuple(shape)
        return cls(
            shape=shape,
            dtype=dtype,
            layout=Layout(shape),
            storage=expr.StorageKind.UMAT,
        )

    @classmethod
    def from_shape(
        cls,
        shape: tuple,
        dtype: expr.DType = expr.DType.f32,
        storage: "str | expr.StorageKind" = "gmem",
        layout: object = None,
    ) -> TensorType:
        """A plain (unsharded) tensor of *shape*; its layout defaults to the compact one."""
        return cls(
            shape=tuple(shape),
            dtype=dtype,
            layout=Layout(tuple(shape)) if layout is None else layout,
            storage=storage,
        )


class VoidMember:
    """The empty ``TupleType`` of the class that holds it, built on first access and then shared."""

    def __init__(self) -> None:
        """Start with no value; the owner class is not complete yet."""
        self.value: TupleType | None = None

    def __get__(self, instance: object, owner: type[TupleType]) -> TupleType:
        """The one empty tuple type this member stands for."""
        if self.value is None:
            self.value = owner(())
        return self.value


@dataclass(frozen=True)
class TupleType:
    """An ordered group of values.

    ``TupleType.Void`` has no fields: it is the ``Call.type`` of an effect Op that
    has no result value (``Copy``, ``Fill``, ``Mma``), placed in Stmt position via
    ``Evaluate(op, args)``.
    """

    fields: tuple[Union["TensorType", "TupleType", "CallableType"], ...] = ()
    Void = VoidMember()


Type = Union[TensorType, TupleType, "CallableType", "PointerType"]
