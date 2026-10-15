from __future__ import annotations

from dataclasses import replace
from functools import cache
from math import prod

from tilefoundry.evaluator.registry import register_eval
from tilefoundry.evaluator.value import TensorValue
from tilefoundry.ir.core import Call, Op
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.pattern import is_ranked_tensor
from tilefoundry.ir.types import Broadcast, TensorType
from tilefoundry.ir.types.dim import DimMul, simplify_dim
from tilefoundry.ir.types.layout import Layout, LayoutBase, resolve_layout
from tilefoundry.ir.types.shard_layout import (
    ShardLayout,
    Split,
    shard_layout_local_shape,
    shard_layout_of,
)
from tilefoundry.ir.types.storage import StorageKind
from tilefoundry.ir.types.stride import compact_row_major
from tilefoundry.ir.types.utils import is_literal_shape, strip_broadcast_layout
from tilefoundry.visitor_registry import register_typeinfer
from tilefoundry.visitor_registry.access_relation import (
    AccessRelation,
    identity_access,
    iterating,
    register_access_relation,
)
from tilefoundry.visitor_registry.buffer_alias import register_buffer_alias


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


def _materialize_reshard_strides(
    layout: ShardLayout,
    src_ty: TensorType,
    new_storage: StorageKind,
) -> ShardLayout:
    """Materialize the sugar-default ``layout.layout.strides``.

    Pre-condition: ``layout.layout.strides is None`` (sugar). For
    verbose paths the caller short-circuits before this helper.
    UMAT materializes in RMEM, so it uses that boundary when choosing sugar
    strides.
    """
    src_storage = StorageKind.RMEM if src_ty.storage is StorageKind.UMAT else src_ty.storage
    src_layout = src_ty.layout if isinstance(src_ty.layout, ShardLayout) else None
    if src_storage == new_storage:
        if _src_form_is_per_instance(src_layout):
            new_strides = _per_instance_strides(layout)
        else:
            new_strides = _shared_engine_strides(layout)
    else:
        src_lvl = _storage_level(src_storage)
        dst_lvl = _storage_level(new_storage)
        if dst_lvl > src_lvl:
            new_strides = _shared_engine_strides(layout)
        else:
            new_strides = _per_instance_strides(layout)
    return ShardLayout(
        layout=Layout(shape=layout.layout.shape, strides=new_strides),
        attrs=layout.attrs,
        mesh=layout.mesh,
    )


@register_op
class Reshard(Op):
    """Convert *x* to a target layout / storage in place, preserving the logical shape."""

    scope_inputs = ()

    x = ParamDef(kind="input", pattern=is_ranked_tensor())
    layout = ParamDef(kind="attribute", annotation=LayoutBase, default=None)
    storage = ParamDef(kind="attribute", default=None)


@cache
def _same_locations(source: TensorType, result: TensorType) -> bool:
    """Every element must retain its complete set of storage instances and addresses."""
    from tilefoundry.ir.hir.tensor.bitcast import _address_points  # noqa: PLC0415
    from tilefoundry.ir.types.int_tuple import flatten  # noqa: PLC0415

    def locations(type_, other):
        shard = shard_layout_of(type_.layout)
        mesh = shard.mesh if shard is not None else getattr(shard_layout_of(other.layout), "mesh", None)
        layout = strip_broadcast_layout(type_.layout)
        if mesh is not None and not isinstance(layout, ShardLayout):
            layout = ShardLayout(layout, (Broadcast(),) * len(flatten(mesh.layout.shape)), mesh)
        ignored = {StorageKind.GMEM: ("cta", "thread"), StorageKind.SMEM: ("thread",)}.get(type_.storage, ())
        points = [set() for _ in range(prod(type_.shape))]
        for logical, unit, full, local in _address_points(layout, len(points), type_.shape):
            instance = tuple(pair for pair in unit if pair[0] not in ignored)
            points[logical].add((instance, local if type_.storage is StorageKind.RMEM else full))
        return points

    try:
        if source.layout == result.layout:
            return True
        if source.storage is StorageKind.GMEM and all(
            topology.name in ("cta", "thread")
            for type_ in (source, result)
            for topology in getattr(getattr(shard_layout_of(type_.layout), "mesh", None), "topologies", ())
        ):
            before = getattr(shard_layout_of(source.layout), "layout", source.layout)
            after = getattr(shard_layout_of(result.layout), "layout", result.layout)
            if before == after:
                return True
        if all(shard_layout_of(type_.layout) is None for type_ in (source, result)):
            return all(
                before[2] == after[2]
                for before, after in zip(
                    _address_points(source.layout, prod(source.shape), source.shape),
                    _address_points(result.layout, prod(result.shape), result.shape), strict=True,
                )
            )
        return locations(source, result) == locations(result, source)
    except ValueError:
        return False


@register_buffer_alias(Reshard)
def _buffer_alias(call: Call) -> int | None:
    source, result = call.args[0].type, call.type
    if result.storage is not source.storage or not is_literal_shape(result.shape):
        return None
    source = replace(source, layout=resolve_layout(source.layout, source.shape))
    result = replace(result, layout=resolve_layout(result.layout, result.shape))
    return 0 if _same_locations(source, result) else None


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
    if (
        isinstance(new_layout, ShardLayout)
        and isinstance(new_layout.layout, Layout)
        and new_layout.layout.strides is None
    ):
        new_layout = _materialize_reshard_strides(new_layout, x_ty, new_storage)
    return TensorType(
        shape=x_ty.shape,
        dtype=x_ty.dtype,
        layout=new_layout,
        storage=new_storage,
    )


@register_eval(Reshard)
def _eval_reshard(ctx):

    return TensorValue(data=ctx.args[0].data, type=ctx.result_type)
