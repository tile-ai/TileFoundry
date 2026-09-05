"""Effect-form TIR Ops for the CUDA warp-scoped exchange primitives.

These ops are **defined but not lowered**: there is no CUDA codegen emit for any
of them. The runtime implementations they name live at
``include/tilefoundry/runtime/cuda/ops/warp.cuh`` and are exercised directly by
``tests/integration/test_runtime_device_ops.py``; wiring the emit belongs with
the TIR backend rebuild, and an emit written now would be rewritten by it while
these definitions would not.

See [tir §2.3](docs/spec/tir.md#23-tir-ops).
"""

from __future__ import annotations

from enum import Enum

from tilefoundry.ir.core import Op
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.pattern import Scalar
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.types import UnitType
from tilefoundry.ir.types.storage import StorageKind
from tilefoundry.visitor_registry import register_typeinfer, register_verify_stmt

__all__ = ["ShuffleXor", "ShuffleElect", "WarpReduceKind", "WarpReduce"]

_WARP = 32


def _require_rmem(ctx, call, at: int, who: str, role: str) -> None:
    """Report unless operand *at* is register-resident.

    ``_WARP`` above is the lane count a shuffle's partner is chosen inside, so
    every constraint here is stated against it rather than against a mesh.

    A warp shuffle moves a value between lanes' registers. An operand in shared
    or global memory is not wrong by a tolerance -- it names a different
    instruction -- so this is an error rather than a note.
    """
    ty = ctx.type_of(call.args[at])
    if ty.storage != StorageKind.RMEM:
        ctx.error(call, f"{who} {role} must be rmem, got {ty.storage}")


@register_op(dialect="T", category="warp", name="shuffle_xor")
class ShuffleXor(Op):
    """Exchange a value with the lane whose id differs by ``lane_mask``.

    ``dst`` receives the ``src`` of lane ``laneid ^ lane_mask``. One
    ``__shfl_xor_sync``; with masks 1, 2, 4, 8, 16 in turn it is the butterfly
    ``WarpReduce`` is built from.
    """

    src = ParamDef(kind="input", pattern=Scalar)
    dst = ParamDef(kind="input", pattern=Scalar)
    lane_mask = ParamDef(kind="attribute", annotation=int)


@register_typeinfer(ShuffleXor)
def _(call: "Call", ctx: "TypeInferContext") -> UnitType:
    return UnitType()


@register_verify_stmt(ShuffleXor)
def _(call: "Call", ctx: "VerifyContext") -> None:
    op = call.target
    mask = op.lane_mask
    if not isinstance(mask, int) or not 0 < mask < _WARP:
        ctx.error(call, f"ShuffleXor.lane_mask must be in 1..{_WARP - 1}, got {mask!r}")
    _require_rmem(ctx, call, 0, "ShuffleXor", "src")
    _require_rmem(ctx, call, 1, "ShuffleXor", "dst")
    src = ctx.type_of(call.args[0])
    dst = ctx.type_of(call.args[1])
    if src.dtype != dst.dtype:
        ctx.error(call, f"ShuffleXor dtype mismatch: {src.dtype} vs {dst.dtype}")


@register_op(dialect="T", category="warp", name="shuffle_elect")
class ShuffleElect(Op):
    """Select exactly one thread of the leading ``width`` threads of the CTA.

    ``dst`` is true on that one thread and false on every other. It is one
    thread of the *block*, not one lane of each warp: an mbarrier armed for a
    single arrival is only correct under the first reading.
    """

    dst = ParamDef(kind="input", pattern=Scalar)
    width = ParamDef(kind="attribute", annotation=int)


@register_typeinfer(ShuffleElect)
def _(call: "Call", ctx: "TypeInferContext") -> UnitType:
    return UnitType()


@register_verify_stmt(ShuffleElect)
def _(call: "Call", ctx: "VerifyContext") -> None:
    width = call.target.width
    if not isinstance(width, int) or width <= 0:
        ctx.error(call, f"ShuffleElect.width must be a positive int, got {width!r}")
    _require_rmem(ctx, call, 0, "ShuffleElect", "dst")


class WarpReduceKind(Enum):
    """Which combine a ``WarpReduce`` folds with."""

    SUM = "sum"
    MAX = "max"
    MIN = "min"


@register_op(dialect="T", category="warp", name="warp_reduce")
class WarpReduce(Op):
    """Fold ``src`` across the warp's 32 lanes, leaving the result in every lane.

    Defined as five ``ShuffleXor`` steps with masks 16, 8, 4, 2, 1 and a combine
    between them -- five shuffles and five combines, not a loop over 32 lanes.
    The count is stated here rather than left to a lowering to reveal, because
    it is the reason a caller reaches for this instead of a shared-memory fold.

    ``kind`` carries the combine, the way ``Binary`` and ``Unary`` carry theirs;
    a class per combine would differ from its siblings in nothing else.
    """

    src = ParamDef(kind="input", pattern=Scalar)
    dst = ParamDef(kind="input", pattern=Scalar)
    kind = ParamDef(kind="attribute", annotation=WarpReduceKind)


@register_typeinfer(WarpReduce)
def _(call: "Call", ctx: "TypeInferContext") -> UnitType:
    return UnitType()


@register_verify_stmt(WarpReduce)
def _(call: "Call", ctx: "VerifyContext") -> None:
    op = call.target
    if not isinstance(op.kind, WarpReduceKind):
        ctx.error(call, f"WarpReduce: kind must be WarpReduceKind, got {type(op.kind)}")
    _require_rmem(ctx, call, 0, "WarpReduce", "src")
    _require_rmem(ctx, call, 1, "WarpReduce", "dst")
    src = ctx.type_of(call.args[0])
    dst = ctx.type_of(call.args[1])
    if src.dtype != dst.dtype:
        ctx.error(call, f"WarpReduce dtype mismatch: {src.dtype} vs {dst.dtype}")
