"""Codegen for the generic Binary and Unary TIR effect-form Ops."""

from __future__ import annotations

from types import MappingProxyType

from tilefoundry import codegen
from tilefoundry.ir import expr, tir, types, utils
from tilefoundry_nvidia.cuda import target
from tilefoundry_nvidia.cuda.codegen import context

from .cute_layout import render_layout_value

BINARY_TAG = MappingProxyType({
    expr.BinaryKind.MUL: "tilefoundry::primitive::mul_op",
    expr.BinaryKind.ADD: "tilefoundry::primitive::add_op",
    expr.BinaryKind.SUB: "tilefoundry::primitive::sub_op",
    expr.BinaryKind.DIV: "tilefoundry::primitive::div_op",
})

UNARY_TAG = MappingProxyType({
    expr.UnaryKind.RSQRT: "tilefoundry::primitive::rsqrt_op",
    expr.UnaryKind.NEG: "tilefoundry::primitive::neg_op",
    expr.UnaryKind.RELU: "tilefoundry::primitive::relu_op",
    expr.UnaryKind.SQUARE: "tilefoundry::primitive::square_op",
    expr.UnaryKind.CAST: "tilefoundry::primitive::identity_op",
})


def materialised_shape(ty) -> tuple:
    """Return the cute-side materialised shape for *ty*.

    Sharded Tensors are materialised at the ``ShardLayout``'s
    per-mesh-position local shape, not the logical
    ``TensorType.shape``. Elementwise helpers must iterate by the
    local count or they walk past the per-thread / per-CTA
    allocation.
    See [shard §7.1.1](docs/spec/shard.md#711-layoutshape).
    """
    layout = getattr(ty, "layout", None)
    if isinstance(layout, types.ShardLayout):

        return types.shard_layout_local_shape(layout)
    return utils.shape_upper_bound(ty.shape)


def materialised_shape_dyn(ty) -> tuple:
    """Materialised shape dyn.

    Like ``materialised_shape`` but preserves ``DimVar`` entries so
    the caller can ask for a runtime (string) iteration count.
    See [shard §7.1.1](docs/spec/shard.md#711-layoutshape).
    """
    layout = getattr(ty, "layout", None)
    if isinstance(layout, types.ShardLayout):
        return types.shard_layout_local_shape(layout)
    return tuple(ty.shape)


def runtime_total(ty, ctx: context.Context) -> object:
    """Runtime element count for *ty* — int or C++ expression string.

    Uses the kernel's DimVar runtime scalars registered by the
    PrimFunction emitter so dynamic dims drive loop counts at launch
    time instead of the static envelope upper bound.
    """
    return utils.shape_runtime_total(materialised_shape_dyn(ty), ctx.dynamic_extents)


def tensor_expr(var, ctx: context.Context) -> str:
    """Tensor expr.

    Kernel-param tensor operands are accessed through the cute wrap
    (``<name>_tensor``) the PrimFunction emitter materialises at the
    top of the body; non-param vars are referenced directly.
    """
    base = ctx.name_for(var)
    return f"{base}_tensor" if ctx.is_kernel_param(var) else base


def extent_entry(e) -> str:
    """A cute shape entry: static extents stay in the type, dynamic ones do not."""
    return f"cute::Int<{int(e)}>{{}}" if isinstance(e, int) else str(e)


def stride_entry(e) -> str:
    """A cute stride entry: broadcast modes are always static."""
    return f"cute::Int<{int(e)}>{{}}"


def broadcast_modes(dst_shape, rhs_shape, n_dst_runtime) -> tuple | None:
    """The destination-shaped layout *rhs* is read through, or ``None``.

    Every shape relation the retired ``binary_bcast_*`` family named is one
    ``(shape, stride)`` pair over the destination's domain -- 1 on an axis
    ``rhs`` supplies, 0 on one it does not. ``None`` means it already spans the
    domain. The tests keep the old family's order, so the same programs match
    the same way; ``cell`` and ``col`` land on one layout because they were one
    relation written twice. *n_dst_runtime* may be a C++ expression: a stride-0
    mode does not care what its extent is.
    """
    n_dst = utils.shape_numel_upper_bound(dst_shape) if dst_shape else 1
    n_rhs = utils.shape_numel_upper_bound(rhs_shape) if rhs_shape else 1
    if n_rhs >= n_dst:
        return None
    if n_rhs == 1:
        return (n_dst_runtime,), (0,)
    if (
        len(dst_shape) == 2
        and len(rhs_shape) == 2
        and rhs_shape[-1] == 1
        and dst_shape[0] == rhs_shape[0]
    ):
        return tuple(int(d) for d in dst_shape), (1, 0)
    if len(dst_shape) == 2 and len(rhs_shape) == 1 and dst_shape[-1] == rhs_shape[0]:
        return tuple(int(d) for d in dst_shape), (0, 1)
    if n_dst % n_rhs == 0:
        return (n_rhs, n_dst // n_rhs), (1, 0)
    return None


def broadcast_view(rhs_n: str, modes, ctx: context.Context) -> str:
    """Bind *rhs_n* read through a stride-0 layout, and name the binding."""
    shape, stride = modes
    name = f"bcast_{ctx.next_id()}"
    ctx.emit(
        f"auto {name} = tilefoundry::local_tensor({rhs_n}).compose("
        f"{render_layout_value(shape, stride, extent_entry, stride_entry)});"
    )
    return name


def dyn_clip(dst, operands, ctx: context.Context) -> str | None:
    """The run-time count a plain dynamic operand set has to be clipped to."""
    if any(isinstance(getattr(v.type, "layout", None), types.ShardLayout)
           for v in (dst, *operands)):
        return None
    if not expr.shape_has_dim_var(dst.type.shape):
        return None
    return str(utils.shape_runtime_total(dst.type.shape, ctx.dynamic_extents))


def binary_fn(tag: str, dst_t: str, in_ts: tuple[str, ...]) -> str:
    """The ``fn`` a binary kind calls with: the op tag, or a lambda."""
    if len({dst_t, *in_ts}) <= 1:
        return f"{tag}{{}}"
    return (
        f"[](auto a, auto b) {{ return {tag}{{}}({dst_t}(a), {dst_t}(b)); }}"
    )


def cpp_dtype(ty, ctx: context.Context) -> str:
    return ctx.dtype_to_cpp(ty.dtype.name)


@codegen.register(target.Target, codegen.Role.EMIT, tir.arith.Binary)
def emit_binary(call, ctx: context.Context) -> None:
    lhs, rhs, dst = call.args
    lhs_n = tensor_expr(lhs, ctx)
    rhs_n = tensor_expr(rhs, ctx)
    dst_n = tensor_expr(dst, ctx)

    lhs_modes = broadcast_modes(
        materialised_shape(dst.type),
        materialised_shape(lhs.type),
        runtime_total(dst.type, ctx),
    )
    if lhs_modes is not None:
        lhs_n = broadcast_view(lhs_n, lhs_modes, ctx)

    modes = broadcast_modes(
        materialised_shape(dst.type),
        materialised_shape(rhs.type),
        runtime_total(dst.type, ctx),
    )
    if modes is not None:
        rhs_n = broadcast_view(rhs_n, modes, ctx)

    fn = binary_fn(
        BINARY_TAG[call.target.kind],
        cpp_dtype(dst.type, ctx),
        (cpp_dtype(lhs.type, ctx), cpp_dtype(rhs.type, ctx)),
    )
    n = dyn_clip(dst, (lhs, rhs), ctx)
    if n is None:
        ctx.emit(f"tilefoundry::ops::elementwise({dst_n}, {fn}, {lhs_n}, {rhs_n});")
        return
    ctx.emit("{")
    ctx.indent()
    ctx.emit(f"auto tilefoundry_elementwise_n = cute::make_layout({n});")
    for view, base in (("tilefoundry_elementwise_dst", dst_n), ("tilefoundry_elementwise_a", lhs_n), ("tilefoundry_elementwise_b", rhs_n)):
        ctx.emit(f"auto {view} = cute::make_tensor({base}.data(), tilefoundry_elementwise_n);")
    ctx.emit(f"tilefoundry::ops::elementwise(tilefoundry_elementwise_dst, {fn}, tilefoundry_elementwise_a, tilefoundry_elementwise_b);")
    ctx.dedent()
    ctx.emit("}")


@codegen.register(target.Target, codegen.Role.EMIT, tir.arith.Unary)
@codegen.register(target.Target, codegen.Role.EMIT, tir.cast.Cast)
def emit_unary(call, ctx: context.Context) -> None:
    """One source, always the bare tag -- a unary needs no lambda.

    An effect-form ``Cast`` is ``UnaryKind.CAST``: the store into the
    destination converts, so its functor is the identity.
    """
    src, dst = call.args
    src_n = tensor_expr(src, ctx)
    dst_n = tensor_expr(dst, ctx)
    kind = expr.UnaryKind.CAST if isinstance(call.target, tir.cast.Cast) else call.target.kind
    tag = UNARY_TAG[kind]
    n = dyn_clip(dst, (src,), ctx)
    if n is None:
        ctx.emit(f"tilefoundry::ops::elementwise({dst_n}, {tag}{{}}, {src_n});")
        return
    ctx.emit("{")
    ctx.indent()
    ctx.emit(f"auto tilefoundry_elementwise_n = cute::make_layout({n});")
    ctx.emit(f"auto tilefoundry_elementwise_dst = cute::make_tensor({dst_n}.data(), tilefoundry_elementwise_n);")
    ctx.emit(f"auto tilefoundry_elementwise_src = cute::make_tensor({src_n}.data(), tilefoundry_elementwise_n);")
    ctx.emit(f"tilefoundry::ops::elementwise(tilefoundry_elementwise_dst, {tag}{{}}, tilefoundry_elementwise_src);")
    ctx.dedent()
    ctx.emit("}")


__all__: list[str] = []
