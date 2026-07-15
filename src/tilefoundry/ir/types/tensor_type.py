from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

from tilefoundry.ir.target.storage import StorageKind, resolve_storage

from .dtype import DType
from .shape_dim import ShapeDim
from .shard import Layout, Mesh, ShardLayout, Split


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
    layout: object
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
        layout: object = None,
        storage: Optional[StorageKind] = StorageKind.RMEM,
    ) -> "TensorType":
        return TensorType(shape=(), dtype=dtype, layout=layout, storage=storage)


def _layout_c_order_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    strides = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    return tuple(strides)


def make_shard_tensor_type(
    shape: tuple,
    dtype: DType = DType.f32,
    storage: "str | StorageKind | None" = "gmem",
    mesh: Optional[Mesh] = None,
    attrs: tuple = (),
) -> "TensorType":
    """Build the canonical sharded ``TensorType`` (``docs/spec/shard.md``
    §7.1.1) from a logical description: ``shape`` is the logical tensor
    shape, ``attrs`` is one entry per mesh axis (``Split(logical_axis)`` /
    ``Broadcast()`` / ``Partial(reduction)``).

    Each logical axis split by one or more mesh axes is factored, in mesh-axis
    order, into one position per splitting mesh axis (sized to that axis's
    extent) plus a residual position (``logical_size // Π(extents)``, omitted
    when 1, an error when the division is not exact) — the internal ``Layout``
    this produces has every ``Split``-bound position sized exactly to its mesh
    extent, which is the §7.1.1 canonical form. ``Split`` attrs are remapped
    from the logical axis to that position. ``mesh=None`` / ``attrs=()``
    yields a plain (unsharded) ``TensorType``.
    """
    shape = tuple(shape)
    if mesh is None or not attrs:
        return TensorType(shape=shape, dtype=dtype, layout=None, storage=storage)

    mesh_shape = mesh.layout.shape
    bindings: dict[int, list[int]] = {}
    for mesh_axis, attr in enumerate(attrs):
        if isinstance(attr, Split):
            bindings.setdefault(attr.axis, []).append(mesh_axis)

    layout_shape: list[int] = []
    factor_position: dict[int, int] = {}
    for logical_axis, axis_size in enumerate(shape):
        splitting_mesh_axes = bindings.get(logical_axis, [])
        if not splitting_mesh_axes:
            layout_shape.append(axis_size)
            continue
        extent_product = 1
        for mesh_axis in splitting_mesh_axes:
            extent = mesh_shape[mesh_axis]
            factor_position[mesh_axis] = len(layout_shape)
            layout_shape.append(extent)
            extent_product *= extent
        if axis_size % extent_product != 0:
            raise ValueError(
                f"make_shard_tensor_type: logical axis {logical_axis} size "
                f"{axis_size} is not divisible by mesh extent product "
                f"{extent_product}"
            )
        residual = axis_size // extent_product
        if residual != 1:
            layout_shape.append(residual)

    remapped_attrs = tuple(
        Split(factor_position[mesh_axis]) if isinstance(attr, Split) else attr
        for mesh_axis, attr in enumerate(attrs)
    )
    layout_shape = tuple(layout_shape)
    layout = ShardLayout(
        layout=Layout(shape=layout_shape, strides=_layout_c_order_strides(layout_shape)),
        attrs=remapped_attrs,
        mesh=mesh,
    )
    return TensorType(shape=shape, dtype=dtype, layout=layout, storage=storage)


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
