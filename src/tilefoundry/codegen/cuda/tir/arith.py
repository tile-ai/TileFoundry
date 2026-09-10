"""Codegen for the generic Binary and Unary TIR effect-form Ops."""

from __future__ import annotations

from tilefoundry.codegen.cuda.context import CodegenContext, register_codegen_cuda
from tilefoundry.ir.tir.arith import Binary, BinaryKind, Unary, UnaryKind
from tilefoundry.ir.types.shape_helpers import (
    shape_has_dim_var,
    shape_numel_upper_bound,
    shape_runtime_total,
    shape_upper_bound,
)
from tilefoundry.ir.types.shard.shard_layout import ShardLayout, shard_layout_local_shape

_BINARY_TAG = {
    BinaryKind.MUL: "tilefoundry::ops::mul_op",
    BinaryKind.ADD: "tilefoundry::ops::add_op",
    BinaryKind.SUB: "tilefoundry::ops::sub_op",
    BinaryKind.DIV: "tilefoundry::ops::div_op",
}

_UNARY_TAG = {
    UnaryKind.RSQRT: "tilefoundry::ops::rsqrt_op",
    UnaryKind.NEG: "tilefoundry::ops::neg_op",
    UnaryKind.RELU: "tilefoundry::ops::relu_op",
    UnaryKind.SQUARE: "tilefoundry::ops::square_op",
    UnaryKind.CAST: "tilefoundry::ops::identity_op",
}


def _materialised_shape(ty) -> tuple:
    """Return the cute-side materialised shape for *ty*.

    Sharded Tensors are materialised at the ``ShardLayout``'s
    per-mesh-position local shape, not the logical
    ``TensorType.shape``. Elementwise helpers must iterate by the
    local count or they walk past the per-thread / per-CTA
    allocation.
    See [shard §7.1.1](docs/spec/shard.md#711-layoutshape).
    """
    layout = getattr(ty, "layout", None)
    if isinstance(layout, ShardLayout):

        return shard_layout_local_shape(layout)
    return shape_upper_bound(ty.shape)


def _materialised_shape_dyn(ty) -> tuple:
    """Materialised shape dyn.

    Like ``_materialised_shape`` but preserves ``DimVar`` entries so
    the caller can ask for a runtime (string) iteration count.
    See [shard §7.1.1](docs/spec/shard.md#711-layoutshape).
    """
    layout = getattr(ty, "layout", None)
    if isinstance(layout, ShardLayout):
        return shard_layout_local_shape(layout)
    return tuple(ty.shape)


def _runtime_total(ty, ctx: CodegenContext) -> object:
    """Runtime element count for *ty* — int or C++ expression string.

    Uses the kernel's DimVar runtime scalars registered by the
    PrimFunction emitter so dynamic dims drive loop counts at launch
    time instead of the static envelope upper bound.
    """
    return shape_runtime_total(_materialised_shape_dyn(ty), ctx._dim_var_runtime)


def _tensor_expr(var, ctx: CodegenContext) -> str:
    """Tensor expr.

    Kernel-param tensor operands are accessed through the cute wrap
    (``<name>_tensor``) the PrimFunction emitter materialises at the
    top of the body; non-param vars are referenced directly.
    """
    base = ctx.name_for(var)
    return f"{base}_tensor" if ctx.is_kernel_param(var) else base


def _extent(e) -> str:
    """A cute shape entry: static extents stay in the type, dynamic ones do not."""
    return f"cute::Int<{int(e)}>{{}}" if isinstance(e, int) else str(e)


def _broadcast_modes(dst_shape, rhs_shape, n_dst_runtime) -> tuple | None:
    """The destination-shaped layout *rhs* is read through, or ``None``.

    Every shape relation the retired ``binary_bcast_*`` family named is one
    ``(shape, stride)`` pair over the destination's domain -- 1 on an axis
    ``rhs`` supplies, 0 on one it does not. ``None`` means it already spans the
    domain. The tests keep the old family's order, so the same programs match
    the same way; ``cell`` and ``col`` land on one layout because they were one
    relation written twice. *n_dst_runtime* may be a C++ expression: a stride-0
    mode does not care what its extent is.
    """
    n_dst = shape_numel_upper_bound(dst_shape) if dst_shape else 1
    n_rhs = shape_numel_upper_bound(rhs_shape) if rhs_shape else 1
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


def broadcast_view(rhs_n: str, modes, ctx: CodegenContext) -> str:
    """Bind *rhs_n* read through a stride-0 layout, and name the binding."""
    shape, stride = modes
    ctx._counter += 1
    name = f"bcast_{ctx._counter}"
    shape_args = ", ".join(_extent(s) for s in shape)
    stride_args = ", ".join(f"cute::Int<{int(s)}>{{}}" for s in stride)
    ctx.emit(
        f"auto {name} = tilefoundry::ops::detail::local_tensor({rhs_n}).compose("
        f"cute::make_layout(cute::make_shape({shape_args}), "
        f"cute::make_stride({stride_args})));"
    )
    return name


def _dyn_clip(dst, operands, ctx: CodegenContext) -> str | None:
    """The run-time count a plain dynamic operand set has to be clipped to."""
    if any(isinstance(getattr(v.type, "layout", None), ShardLayout)
           for v in (dst, *operands)):
        return None
    if not shape_has_dim_var(dst.type.shape):
        return None
    return str(shape_runtime_total(dst.type.shape, ctx._dim_var_runtime))


def _binary_fn(tag: str, dst_t: str, in_ts: tuple[str, ...]) -> str:
    """The ``fn`` a binary kind calls with: the op tag, or a lambda."""
    if len({dst_t, *in_ts}) <= 1:
        return f"{tag}{{}}"
    return (
        f"[](auto a, auto b) {{ return {tag}{{}}({dst_t}(a), {dst_t}(b)); }}"
    )


def _cpp_dtype(ty, ctx: CodegenContext) -> str:
    return ctx.dtype_to_cpp(ty.dtype.name)


@register_codegen_cuda(Binary)
def _emit_binary(call, ctx: CodegenContext) -> None:
    lhs, rhs, dst = call.args
    lhs_n = _tensor_expr(lhs, ctx)
    rhs_n = _tensor_expr(rhs, ctx)
    dst_n = _tensor_expr(dst, ctx)

    modes = _broadcast_modes(
        _materialised_shape(dst.type),
        _materialised_shape(rhs.type),
        _runtime_total(dst.type, ctx),
    )
    if modes is not None:
        rhs_n = broadcast_view(rhs_n, modes, ctx)

    fn = _binary_fn(
        _BINARY_TAG[call.target.kind],
        _cpp_dtype(dst.type, ctx),
        (_cpp_dtype(lhs.type, ctx), _cpp_dtype(rhs.type, ctx)),
    )
    n = _dyn_clip(dst, (lhs, rhs), ctx)
    if n is None:
        ctx.emit(f"tilefoundry::ops::elementwise({dst_n}, {fn}, {lhs_n}, {rhs_n});")
        return
    ctx.emit("{")
    ctx.indent()
    ctx.emit(f"auto tf_ew_n = cute::make_layout({n});")
    for view, base in (("tf_ew_dst", dst_n), ("tf_ew_a", lhs_n), ("tf_ew_b", rhs_n)):
        ctx.emit(f"auto {view} = cute::make_tensor({base}.data(), tf_ew_n);")
    ctx.emit(f"tilefoundry::ops::elementwise(tf_ew_dst, {fn}, tf_ew_a, tf_ew_b);")
    ctx.dedent()
    ctx.emit("}")


@register_codegen_cuda(Unary)
def _emit_unary(call, ctx: CodegenContext) -> None:
    """One source, always the bare tag -- a unary needs no lambda."""
    src, dst = call.args
    src_n = _tensor_expr(src, ctx)
    dst_n = _tensor_expr(dst, ctx)
    tag = _UNARY_TAG[call.target.kind]
    n = _dyn_clip(dst, (src,), ctx)
    if n is None:
        ctx.emit(f"tilefoundry::ops::elementwise({dst_n}, {tag}{{}}, {src_n});")
        return
    ctx.emit("{")
    ctx.indent()
    ctx.emit(f"auto tf_ew_n = cute::make_layout({n});")
    ctx.emit(f"auto tf_ew_dst = cute::make_tensor({dst_n}.data(), tf_ew_n);")
    ctx.emit(f"auto tf_ew_src = cute::make_tensor({src_n}.data(), tf_ew_n);")
    ctx.emit(f"tilefoundry::ops::elementwise(tf_ew_dst, {tag}{{}}, tf_ew_src);")
    ctx.dedent()
    ctx.emit("}")
