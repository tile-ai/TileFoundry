from __future__ import annotations

from dataclasses import replace
from itertools import zip_longest
from math import prod

from tilefoundry.evaluator.registry import register_eval
from tilefoundry.evaluator.value import TensorValue
from tilefoundry.ir.core import Call, Op
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.pattern import is_ranked_tensor
from tilefoundry.ir.types import ComposedLayout, TensorType
from tilefoundry.ir.types.dim import DimMul, simplify_dim
from tilefoundry.ir.types.layout import Layout, LayoutBase, flatten, resolve_layout
from tilefoundry.ir.types.layout_algebra import coalesce
from tilefoundry.ir.types.layout_correspondence import layout_correspondence
from tilefoundry.ir.types.shard_layout import (
    ShardLayout,
    Split,
    shard_layout_of,
)
from tilefoundry.ir.types.storage import StorageKind
from tilefoundry.ir.types.stride import compact_row_major
from tilefoundry.ir.types.utils import is_literal_shape, static_dim_value
from tilefoundry.visitor_registry import register_typeinfer
from tilefoundry.visitor_registry.access_relation import (
    AccessRelation,
    identity_access,
    iterating,
    register_access_relation,
)
from tilefoundry.visitor_registry.buffer_alias import aliased_operand, register_buffer_alias
from tilefoundry.visitor_registry.shard_propagate import partial_reductions_by_axis
from tilefoundry.visitor_registry.type_defaults import default_type


def _dim_mul(a, b):
    """Multiply two shape dims.

    Multiply two shape dims. Folds ``int * int`` to ``int`` (keeping
    static strides static); produces a symbolic dim-expr when either
    operand is a dynamic ``DimVar`` / ``Expr`` (so dynamic shapes do not
    force a premature ``int()`` coercion).
    """
    if isinstance(a, int) and isinstance(b, int):
        return a * b
    return simplify_dim(DimMul, (a, b))


def _c_order_strides(shape: tuple) -> tuple:
    """C-order contiguous strides for *shape*.

    C-order contiguous strides for *shape*. A dynamic axis yields a
    symbolic stride for the axes above it via ``_dim_mul``; static inner
    strides stay plain ints.
    """
    return compact_row_major(shape, mul=_dim_mul)


def _shared_engine_strides(sl: ShardLayout) -> tuple:
    """Shared-engine strides for *sl*.

    Strides are plain C-order over the canonical (unsharded) layout
    shape ``sl.layout.shape``; multiple instances index a single
    underlying engine at disjoint offsets ``i · S[k]``.
    """
    return _c_order_strides(tuple(sl.layout.shape))


def _per_instance_strides(sl: ShardLayout) -> tuple:
    """Per-instance strides for *sl*.

    For each layout dim ``k``:

    - If some mesh axis has ``Split(k)`` → ``S[k] = 0`` (each instance
      gets a distinct engine; the mesh-axis contribution to the
      intra-engine offset is zero).
    - Otherwise → C-order stride over
      ``shard_layout_local_shape(sl)``, with size-1 positions
      normalised to ``0``.
    """
    local_shape = shard_layout_local_shape(sl, require_static=False)
    n = len(local_shape)
    if n == 0:
        return ()
    split_axes: set[int] = set()
    for attr in sl.attrs:
        if isinstance(attr, Split):
            split_axes.add(int(attr.axis))
    base = _c_order_strides(local_shape)
    out = []
    for k in range(n):
        if k in split_axes:
            out.append(0)
        elif isinstance(local_shape[k], int) and local_shape[k] == 1:
            out.append(0)
        else:
            out.append(base[k])
    return tuple(out)


_STORAGE_LEVEL: dict[StorageKind, int] = {
    StorageKind.RMEM: 0,
    StorageKind.SMEM: 1,
    StorageKind.GMEM: 2,
}


def _storage_level(storage: StorageKind) -> int:
    try:
        return _STORAGE_LEVEL[storage]
    except KeyError as exc:  # pragma: no cover - guard
        raise ValueError(f"unknown storage tier: {storage!r}") from exc


def _src_form_is_per_instance(sl_src: "ShardLayout | None") -> bool:
    """True iff *sl_src* carries the per-instance stride form (every Split axis has stride 0).

    True iff *sl_src* carries the per-instance stride form
    (every Split axis has stride 0). Used to inherit the source's
    form on same-storage sugar reshards.
    """
    if sl_src is None or sl_src.layout.strides is None:
        return False
    split_axes: set[int] = set()
    for attr in sl_src.attrs:
        if isinstance(attr, Split):
            split_axes.add(int(attr.axis))
    if not split_axes:
        return False
    strides = tuple(int(s) for s in sl_src.layout.strides)
    return all(strides[k] == 0 for k in split_axes if k < len(strides))


@register_op
class Reshard(Op):
    """Convert *x* to a target layout / storage in place, preserving the logical shape."""

    scope_inputs = ()

    x = ParamDef(kind="input", pattern=is_ranked_tensor())
    layout = ParamDef(kind="attribute", annotation=LayoutBase, default=None)
    storage = ParamDef(kind="attribute", default=None)


def _view_offset(value):
    """Read plain-view origins from starts rather than the value's layout."""
    from tilefoundry.ir.hir.tensor.slice import Slice, window_layout  # noqa: PLC0415
    from tilefoundry.visitor_registry.contexts import TypeInferContext  # noqa: PLC0415

    physical = value.type.layout
    if isinstance(physical, ShardLayout):
        physical = physical.layout
    if isinstance(physical, ComposedLayout) or not isinstance(value, Call):
        return 0
    position = aliased_operand(value)
    if position is None:
        return 0
    source = value.args[position]
    offset = _view_offset(source)
    if isinstance(value.target, Slice):
        starts = value.args[1].elements
        if all(isinstance(static_dim_value(start), int) for start in starts):
            _, moved = window_layout(
                source.type.layout,
                source.type.shape,
                starts,
                value.target.sizes,
                value.target.strides,
                TypeInferContext(),
            )
            offset += moved
    return offset


@register_buffer_alias(Reshard)
def _buffer_alias(call: Call) -> int | None:
    source, result = call.args[0].type, call.type
    if result.storage is not source.storage or not is_literal_shape(result.shape):
        return None
    if any(
        before != after
        for before, after in zip_longest(
            partial_reductions_by_axis(source.layout),
            partial_reductions_by_axis(result.layout),
        )
    ):
        return None
    offset = _view_offset(call.args[0])
    if offset != 0:
        source = replace(source, layout=ComposedLayout(None, offset, source.layout))
    boxes = []
    for held in (source, result):
        layout = resolve_layout(held.layout, held.shape)
        shard = shard_layout_of(layout)
        physical = shard.layout if shard is not None else layout
        if shard is not None and isinstance(layout, ComposedLayout):
            physical = replace(layout, outer=physical)
        flat = flatten(physical)
        correspondence = layout_correspondence(flat.shape, held.shape)
        axes = correspondence.tensor_axes()
        independent, splits = set(), []
        if shard is not None:
            for mesh_axis, attr in enumerate(shard.attrs):
                if (
                    isinstance(attr, Split)
                    and static_dim_value(tuple(flatten(shard.mesh.layout.shape))[mesh_axis]) != 1
                    and flat.strides[attr.axis] == 0
                ):
                    position = attr.axis
                    group = correspondence.groups[axes[position]]
                    suffix = prod(flat.shape[index] for index in group if index > position)
                    splits.append((mesh_axis, axes[position], suffix, flat.shape[position]))
                    independent.add(position)
        kept = tuple(
            (extent, step)
            for index, (extent, step) in enumerate(zip(flat.shape, flat.strides, strict=True))
            if index not in independent
        )
        shape, strides = zip(*kept) if kept else ((1,), (0,))
        address = coalesce(Layout(tuple(shape), tuple(strides)), major="row")
        if isinstance(physical, ComposedLayout) and (
            physical.inner is not None or physical.offset != 0
        ):
            address = replace(physical, outer=address)
        boxes.append((tuple(splits), address))
    return 0 if boxes[0] == boxes[1] else None


@register_access_relation(Reshard)
def _reshard_access(call: "Call", ctx) -> tuple[AccessRelation, ...]:
    """Every logical index reads itself. Where those bytes go is a separate fact.

    A reshard moves a value between storages or redistributes it across
    positions; neither changes which logical element the result's index came
    from, so the boundary is exactly identity and never opaque. Which positions
    those indices are is the reader's question, not this one's.
    """
    logical = ctx.type_of(call.args[0])
    rank = len(logical.shape)
    reads = identity_access(rank)
    return iterating(
        logical.shape,
        (reads, identity_access(rank)),
    )


@register_typeinfer(Reshard)
def _(call: "Call", ctx: "TypeInferContext") -> TensorType:
    if not call.args:
        ctx.error(call, "missing required input 'x'")
    x_ty = ctx.type_of(call.args[0])
    op = call.target
    if op.storage is StorageKind.UMAT:
        ctx.error(
            call,
            "Reshard: destination storage cannot be unmaterialized (umat); "
            "reshard targets a concrete residency",
        )
    storage_changed = op.storage is not None and op.storage != x_ty.storage
    if op.layout is None:
        if storage_changed:
            ctx.error(
                call,
                "Reshard: storage change requires an explicit `layout=` argument.",
            )
        return TensorType(
            shape=x_ty.shape,
            dtype=x_ty.dtype,
            layout=x_ty.layout,
            storage=x_ty.storage,
        )
    new_storage = op.storage if op.storage is not None else x_ty.storage
    new_layout = op.layout
    return default_type(
        TensorType(
            shape=x_ty.shape,
            dtype=x_ty.dtype,
            layout=new_layout,
            storage=new_storage,
        ),
        target=ctx.target,
    )


@register_eval(Reshard)
def _eval_reshard(ctx):

    return TensorValue(data=ctx.args[0].data, type=ctx.result_type)
