"""Emitter for ``tir.memory.AllocTensor`` (Expr Op) anchored by a LetStmt.

Handler signature: ``(let_stmt: LetStmt, ctx: CudaCodegenContext) -> None``.
Materialises a CuTe tensor bound to the storage class indicated by the
LetStmt's var TensorType.

When the var's layout is a ``ShardLayout``, the emitted CUDA wraps the per-thread
backing cute Tensor in ``tilefoundry::make_shard_tensor(...)``. A non-shard
storage (``gmem`` / ``smem`` / ``rmem``) keeps the plain ``TensorType.shape``
materialisation path.
"""
from __future__ import annotations

from tilefoundry.codegen.cuda.context import CudaCodegenContext
from tilefoundry.codegen.cuda.tir.memory.tensor_view import render_shard_layout_value
from tilefoundry.codegen.cuda.tir.stmts.mesh_scope import program_topologies
from tilefoundry.ir.tir.memory import AllocTensor
from tilefoundry.ir.tir.stmts import LetStmt
from tilefoundry.ir.types.shape_helpers import (
    shape_numel_upper_bound,
    shape_upper_bound,
    upper_bound,
)
from tilefoundry.ir.types.shard.shard_layout import ShardLayout, shard_layout_local_shape
from tilefoundry.ir.types.storage import StorageKind
from tilefoundry.target import CudaTarget
from tilefoundry.visitor_registry.registries import Role, register_codegen


def _emit_plain_alloc(
    ctx: CudaCodegenContext,
    var,
    name: str,
    storage: StorageKind,
    local_shape: tuple,
) -> str:
    """Emit the backing cute tensor and return the identifier for it.

    The caller uses it as the engine of a ``make_shard_tensor`` wrap, or
    directly as the visible name when no wrap is needed.

    Registers get a named array and a pointer engine, never
    ``make_tensor<T>(layout)``: that builds an ``ArrayEngine`` holding its
    storage inside the tensor object, and ``ShardTensor`` keeps its engine by
    value, so wrapping one copies the registers and writes through ``local()``
    land in the copy.
    """
    if storage is StorageKind.UMAT:
        raise ValueError(
            f"AllocTensor for {name!r} has unmaterialized storage (storage=umat); "
            "a memory-resident tensor must carry a concrete StorageKind"
        )
    total = shape_numel_upper_bound(local_shape)
    if len(local_shape) > 1:
        shape_args = ", ".join(
            f"cute::Int<{upper_bound(s)}>" for s in local_shape
        )
        layout = f"cute::make_layout(cute::Shape<{shape_args}>{{}})"
    else:
        layout = f"cute::make_layout(cute::Shape<cute::Int<{total}>>{{}})"
    cpp_type = ctx.dtype_to_cpp(var.type.dtype.name)
    if storage == StorageKind.SMEM:


        ctx.emit(f"__shared__ __align__(16) {cpp_type} {name}_buf[{total}];")
        ctx.emit(
            f"auto {name} = cute::make_tensor("
            f"cute::make_smem_ptr({name}_buf), {layout});"
        )
    elif storage == StorageKind.RMEM:
        ctx.emit(f"__align__(16) {cpp_type} {name}_buf[{total}];")
        ctx.emit(
            f"auto {name} = cute::make_tensor("
            f"cute::make_rmem_ptr({name}_buf), {layout});"
        )
    else:
        ctx.emit(f"auto {name} = cute::make_tensor<{cpp_type}>({layout});")
    return name


@register_codegen(CudaTarget, Role.EMIT, AllocTensor)
def _emit(let: LetStmt, ctx: CudaCodegenContext) -> None:
    """Materialize plain or sharded storage for one allocation.

    Shard layout shapes are global and backing storage uses their derived local
    shape, except for shared memory split across the block's *threads*: shared
    memory belongs to the CTA, ``local()`` offsets into it by the instance's
    own base, and a buffer sized to one instance's share is read past by every
    instance but the first. The local shape stays right for rmem -- a thread's
    registers are its own -- and for a cta-scoped mesh, where each CTA owns its
    slice. See [shard §7.1.1](docs/spec/shard.md#711-layoutshape).
    """
    var = let.var
    name = ctx.name_for(var)
    storage = var.type.storage
    layout_obj = getattr(var.type, "layout", None)

    if isinstance(layout_obj, ShardLayout):






        if (
            storage is StorageKind.SMEM
            and program_topologies(layout_obj.mesh)[0].name == "thread"
        ):
            backing_shape = shape_upper_bound(layout_obj.layout.shape)
        else:
            backing_shape = shard_layout_local_shape(layout_obj)
        local_shape = tuple(s for s in backing_shape if s != 1) or (1,)


        buf_name = f"{name}_buf_t"
        _emit_plain_alloc(ctx, var, buf_name, storage, local_shape)





        global_total = shape_numel_upper_bound(shape_upper_bound(var.type.shape))
        global_layout = (
            f"cute::make_layout(cute::Shape<cute::Int<{global_total}>>{{}})"
        )
        preamble, shard_value = render_shard_layout_value(
            name, layout_obj, ctx.dynamic_extents, storage, ctx
        )
        for line in preamble:
            ctx.emit(line)
        ctx.emit(
            f"auto {name} = tilefoundry::make_shard_tensor("
            f"{buf_name}, {global_layout}, {shard_value});"
        )
    else:



        local_shape = shape_upper_bound(var.type.shape)
        _emit_plain_alloc(ctx, var, name, storage, local_shape)
