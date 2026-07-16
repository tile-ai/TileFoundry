from __future__ import annotations

from tilefoundry.ir.core import Expr, Op
from tilefoundry.ir.core.expr import Call, Constant
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.pattern import Tensor
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.core.registry import register_typeinfer
from tilefoundry.ir.types import DType, TensorType
from tilefoundry.ir.types.dim import DimAdd, DimFloorDiv, DimSub, simplify_dim
from tilefoundry.ir.types.shape_helpers import static_dim_value
from tilefoundry.visitor_registry.shard_propagate import partial_reductions_by_axis


@register_op
class Conv2D(Op):
    input = ParamDef(kind="input", pattern=Tensor)
    weight = ParamDef(kind="input", pattern=Tensor)
    bias = ParamDef(kind="input", pattern=Tensor)
    stride = ParamDef(kind="attribute", annotation=tuple)
    padding = ParamDef(kind="attribute", annotation=tuple)
    dilation = ParamDef(kind="attribute", annotation=tuple)
    groups = ParamDef(kind="attribute", annotation=int)
def _i64(value: int) -> Constant:
    return Constant(type=TensorType.scalar(DType.i64), value=value)

def _as_expr(v):
    if isinstance(v, Expr):
        return v
    return _i64(int(v))

def _out_spatial(in_dim: Expr, k: int, s: int, p: int, d: int) -> Expr:
    """Compute (in + 2*p - d*(k-1) - 1) // s + 1, keeping symbolic dims alive.

    If `in_dim` is a Constant the result is also a Constant; otherwise we
    build a `dim.*` Expr tree so downstream passes can simplify.
    """
    # effective_kernel = d * (k - 1) + 1
    eff_k = d * (k - 1) + 1
    # ``simplify_dim`` collapses all-Constant chains to a single
    # Constant at construction time. The explicit
    # Constant short-circuit above is no longer needed — the
    # bottom-up fold handles it.
    add_pad = simplify_dim(DimAdd, (in_dim, _i64(2 * p)))
    sub_k = simplify_dim(DimSub, (add_pad, _i64(eff_k)))
    div_s = simplify_dim(DimFloorDiv, (sub_k, _i64(s)))
    plus_1 = simplify_dim(DimAdd, (div_s, _i64(1)))
    return plus_1

@register_typeinfer(Conv2D)
def _(call: "Call", ctx: "TypeInferContext") -> TensorType:
    x = ctx.type_of(call.args[0])
    w = ctx.type_of(call.args[1])
    op = call.target
    if x.dtype != w.dtype:
        ctx.error(call, f"Conv2D dtype mismatch: {x.dtype} vs {w.dtype}")
    if len(x.shape) != 4 or len(w.shape) != 4:
        ctx.error(call, "Conv2D expects rank-4 input and weight (NCHW / OIHW)")
    if len(op.stride) != 2 or len(op.padding) != 2 or len(op.dilation) != 2:
        ctx.error(call, "Conv2D stride/padding/dilation must all be length-2")
    sH, sW = op.stride
    pH, pW = op.padding
    dH, dW = op.dilation
    # weight layout: (O, I/groups, kH, kW). kH / kW must be concrete ints.
    kH, kW = static_dim_value(w.shape[2]), static_dim_value(w.shape[3])
    if kH is None or kW is None:
        ctx.error(call, "Conv2D kernel spatial dims (H, W) must be static")
    # A pre-existing Partial(reduction) on either operand (weight replication)
    # propagates only for "sum" — convolution is linear in each operand for
    # the other fixed, but does not preserve order, so max/min never commute.
    for arg, t in (("input", x), ("weight", w)):
        bad = tuple(
            reduction
            for reduction in partial_reductions_by_axis(t.layout)
            if reduction not in (None, "sum")
        )
        if bad:
            ctx.error(
                call,
                f"Conv2D: Partial({sorted(bad)}) input on {arg} is unsound "
                f"(convolution is linear, commutes with sum only) — insert "
                f"reshard({arg}, Broadcast) before this consumer",
            )
    N = x.shape[0]
    C_out = w.shape[0]
    H_out = _out_spatial(x.shape[2], kH, sH, pH, dH)
    W_out = _out_spatial(x.shape[3], kW, sW, pW, dW)
    return TensorType(
        shape=(N, C_out, H_out, W_out),
        dtype=x.dtype,
        layout=x.layout,
        storage=x.storage,
    )
