"""Emitter for ``tir.memory.AllocTensor`` (Expr Op) anchored by a LetStmt.

Handler signature: ``(let_stmt: LetStmt, ctx: CodegenContext) -> None``.
Materialises a CuTe tensor bound to the storage class indicated by the
LetStmt's var TensorType.

When the var's layout is a ``ShardLayout``, the emitted CUDA wraps the per-thread
backing cute Tensor in ``tilefoundry::make_shard_tensor(...)``. A non-shard
storage (``gmem`` / ``smem`` / ``rmem``, or ``storage=None``) keeps the plain
``TensorType.shape`` materialisation path.
"""
from __future__ import annotations

from functools import reduce
from operator import mul

from tilefoundry.codegen.cuda.context import CodegenContext, register_codegen_cuda
from tilefoundry.codegen.cuda.tir.memory.tensor_view import render_shard_layout_value
from tilefoundry.ir.target.storage import StorageKind
from tilefoundry.ir.tir.memory import AllocTensor
from tilefoundry.ir.tir.stmts import LetStmt
from tilefoundry.ir.types.shape_helpers import shape_upper_bound, upper_bound
from tilefoundry.ir.types.shard.shard_layout import ShardLayout, shard_layout_local_shape


def _total(shape) -> int:
    # Dynamic dims (``DimVar``) collapse to their envelope upper bound
    # so the static compile-time element count fits any runtime shape.
    if not shape:
        return 1
    return reduce(mul, (upper_bound(s) for s in shape), 1)


def _emit_plain_alloc(
    ctx: CodegenContext,
    var,
    name: str,
    storage: StorageKind | None,
    local_shape: tuple,
) -> str:
    """Emit the per-thread / per-CTA backing cute tensor and return
    the C++ identifier the caller can use as the engine for a
    ``make_shard_tensor`` wrap (or directly as the visible name when
    no shard wrap is needed)."""
    if storage is None:
        raise ValueError(
            f"AllocTensor for {name!r} has no memory space (storage=None); a "
            f"memory-resident tensor must carry a concrete StorageKind"
        )
    total = _total(local_shape)
    if len(local_shape) > 1:
        shape_args = ", ".join(
            f"cute::Int<{upper_bound(s)}>" for s in local_shape
        )
        layout = f"cute::make_layout(cute::Shape<{shape_args}>{{}})"
    else:
        layout = f"cute::make_layout(cute::Shape<cute::Int<{total}>>{{}})"
    cpp_type = ctx.dtype_to_cpp(var.type.dtype.name)
    if storage == StorageKind.SMEM:
        # 16B-align the shared tile so a 128-bit vectorized / ``cp.async``
        # access into it has an aligned base (alignment only ever increases).
        ctx.emit(f"__shared__ __align__(16) {cpp_type} {name}_buf[{total}];")
        ctx.emit(
            f"auto {name} = cute::make_tensor("
            f"cute::make_smem_ptr({name}_buf), {layout});"
        )
    else:
        ctx.emit(f"auto {name} = cute::make_tensor<{cpp_type}>({layout});")
    return name


@register_codegen_cuda(AllocTensor)
def _emit(let: LetStmt, ctx: CodegenContext) -> None:
    var = let.var
    name = ctx.name_for(var)
    storage = var.type.storage
    layout_obj = getattr(var.type, "layout", None)

    if isinstance(layout_obj, ShardLayout):
        # ShardTensor materialisation: per-thread backing tensor sized
        # to the *derived* per-thread local shape (size-1 dims dropped
        # so the cute rank matches ``local()``'s coalesced view), then
        # wrapped as a ``tilefoundry::ShardTensor`` with the full
        # ShardLayout type.  ``layout.layout.shape`` itself is the
        # **global / unsharded** shape (spec §7).
        local_shape = shard_layout_local_shape(layout_obj)
        local_shape = tuple(s for s in local_shape if s != 1) or (1,)
        # Emit the per-thread cute tensor under a ``_buf`` name —
        # the visible identifier is the ShardTensor wrap.
        buf_name = f"{name}_buf_t"
        _emit_plain_alloc(ctx, var, buf_name, storage, local_shape)
        # Global-view cute layout: a flat 1-D ``cute::Shape<Int<N>>``
        # whose product equals the logical extent. ``local_impl``
        # reads this only to derive the per-thread offset; the actual
        # element access goes through the engine + ShardLayout
        # strides.
        global_total = _total(shape_upper_bound(var.type.shape))
        global_layout = (
            f"cute::make_layout(cute::Shape<cute::Int<{global_total}>>{{}})"
        )
        preamble, shard_value = render_shard_layout_value(
            name, layout_obj, getattr(ctx, "_dim_var_runtime", None)
        )
        for line in preamble:
            ctx.emit(line)
        ctx.emit(
            f"auto {name} = tilefoundry::make_shard_tensor("
            f"{buf_name}, {global_layout}, {shard_value});"
        )
    else:
        # Dynamic dims (``DimVar``) lower to their envelope upper bound
        # for the static cute layout; the runtime shape is plumbed via
        # ``<param>_shape_<axis>`` scalar kernel params separately.
        local_shape = shape_upper_bound(var.type.shape)
        _emit_plain_alloc(ctx, var, name, storage, local_shape)
