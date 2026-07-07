from __future__ import annotations

from tilefoundry.evaluator.registry import register_eval
from tilefoundry.evaluator.value import TensorValue
from tilefoundry.ir.core import Op
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.pattern import Tensor
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.core.registry import register_typeinfer
from tilefoundry.ir.target.storage import StorageKind
from tilefoundry.ir.types import TensorType
from tilefoundry.ir.types.dim import DimMul, simplify_dim
from tilefoundry.ir.types.shard.layout import Layout
from tilefoundry.ir.types.shard.shard_layout import (
    ShardLayout,
    Split,
    shard_layout_local_shape,
)


def _dim_mul(a, b):
    """Multiply two shape dims. Folds ``int * int`` to ``int`` (keeping
    static strides static); produces a symbolic dim-expr when either
    operand is a dynamic ``DimVar`` / ``Expr`` (so dynamic shapes do not
    force a premature ``int()`` coercion)."""
    if isinstance(a, int) and isinstance(b, int):
        return a * b
    return simplify_dim(DimMul, (a, b))

def _c_order_strides(shape: tuple) -> tuple:
    """C-order contiguous strides for *shape*. Used for both shared-
    engine materialization (over canonical global shape) and the
    non-Split fallback path of per-instance materialization (over
    local shape). A dynamic axis yields a symbolic stride for the
    axes above it; static inner strides stay plain ints."""
    if not shape:
        return ()
    strides = [1]
    for d in reversed(shape[1:]):
        strides.insert(0, _dim_mul(strides[0], d))
    return tuple(strides)

def _shared_engine_strides(sl: ShardLayout) -> tuple:
    """Shared-engine strides for *sl*.

    Strides are plain C-order over the canonical (unsharded) cute
    shape ``sl.layout.shape``; multiple instances index a single
    underlying engine at disjoint offsets ``i · S[k]``.
    """
    return _c_order_strides(tuple(sl.layout.shape))

def _per_instance_strides(sl: ShardLayout) -> tuple[int, ...]:
    """Per-instance strides for *sl*.

    For each cute dim ``k``:

    - If some mesh axis has ``Split(k)`` → ``S[k] = 0`` (each instance
      gets a distinct engine; the mesh-axis contribution to the
      intra-engine offset is zero).
    - Otherwise → C-order stride over
      ``shard_layout_local_shape(sl)``, with size-1 positions
      normalised to ``0``.
    """
    local_shape = shard_layout_local_shape(sl)
    n = len(local_shape)
    if n == 0:
        return ()
    split_axes: set[int] = set()
    for attr in sl.attrs:
        if isinstance(attr, Split):
            split_axes.add(int(attr.axis))
    base = [1] * n
    for i in range(n - 2, -1, -1):
        base[i] = base[i + 1] * int(local_shape[i + 1])
    out = []
    for k in range(n):
        if k in split_axes:
            out.append(0)
        elif int(local_shape[k]) == 1:
            out.append(0)
        else:
            out.append(base[k])
    return tuple(out)

# Storage hierarchy. Physical addressability
# ordering; not a free design choice.
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
    """True iff *sl_src* carries the per-instance stride form
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
    """
    src_storage = src_ty.storage
    src_layout = src_ty.layout if isinstance(src_ty.layout, ShardLayout) else None
    if src_storage == new_storage:
        # Same-storage sugar — match the form already present on
        # ``src``. Fall back to shared-engine C-order when ``src``
        # has no ShardLayout (plain kernel-param surface).
        if _src_form_is_per_instance(src_layout):
            new_strides = _per_instance_strides(layout)
        else:
            new_strides = _shared_engine_strides(layout)
    else:
        src_lvl = _storage_level(src_storage)
        dst_lvl = _storage_level(new_storage)
        if dst_lvl > src_lvl:
            # low → high: shared-engine C-order over canonical shape.
            new_strides = _shared_engine_strides(layout)
        else:
            # high → low: per-instance form.
            new_strides = _per_instance_strides(layout)
    return ShardLayout(
        layout=Layout(shape=layout.layout.shape, strides=new_strides),
        attrs=layout.attrs,
        mesh=layout.mesh,
    )

@register_op
class Reshard(Op):
    """Convert *x* to a target layout / storage in place, preserving the logical shape."""
    x = ParamDef(kind="input", pattern=Tensor)
    layout = ParamDef(kind="attribute", annotation=ShardLayout, default=None)
    storage = ParamDef(kind="attribute", default=None)

@register_typeinfer(Reshard)
def _(call: "Call", ctx: "TypeInferContext") -> TensorType:
    if not call.args:
        raise TypeError("Reshard: missing required input 'x'")
    x_ty = ctx.type_of(call.args[0])
    op = call.target
    if op.storage is StorageKind.UMAT:
        ctx.error(
            call,
            "Reshard: destination storage cannot be unmaterialized (umat); "
            "reshard targets a concrete residency",
        )
    if op.layout is not None and not isinstance(op.layout, ShardLayout):
        ctx.error(call, "Reshard.layout must be a ShardLayout (or None to preserve)")
    storage_changed = op.storage is not None and op.storage != x_ty.storage
    if op.layout is None:
        if storage_changed:
            ctx.error(
                call,
                "Reshard: storage change requires an explicit `layout=` "
                "argument.",
            )
        return TensorType(
            shape=x_ty.shape, dtype=x_ty.dtype,
            layout=x_ty.layout, storage=x_ty.storage,
        )
    new_storage = op.storage if op.storage is not None else x_ty.storage
    new_layout = op.layout
    if isinstance(new_layout, ShardLayout) and new_layout.layout.strides is None:
        new_layout = _materialize_reshard_strides(new_layout, x_ty, new_storage)
    return TensorType(
        shape=x_ty.shape, dtype=x_ty.dtype,
        layout=new_layout, storage=new_storage,
    )


@register_eval(Reshard)
def _eval_reshard(ctx):
    # Value-preserving: the logical value is unchanged; only the type's
    # layout / storage are updated (sharding distribution is not executed).
    return TensorValue(data=ctx.args[0].data, type=ctx.result_type)
