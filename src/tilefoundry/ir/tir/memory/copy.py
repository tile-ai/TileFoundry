"""Effect-ful TIR Op ``tir.memory.Copy``.

Copies ``src`` to ``dst``. Memory direction (gmem / smem
/ rmem) is inferred from ``.type.storage`` of each operand. Covers
Load / Store too. The Op is placed in Stmt position as
``Evaluate(Copy, ...)``; the invocation is unit-typed (no result value).
"""

from __future__ import annotations

from tilefoundry.ir.core import Op
from tilefoundry.ir.core.param_def import MemoryEffect, ParamDef
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.pattern import (
    any_threads,
    between_rules,
    moved_tile,
)
from tilefoundry.ir.types import LayoutBase, UnitType
from tilefoundry.ir.types.shard_layout import ShardLayout
from tilefoundry.visitor_registry import register_typeinfer, register_verify_stmt


@register_op
class Copy(Op):
    """Copies ``src`` into ``dst`` (in-place memory write)."""

    src = ParamDef(kind="input", effect=MemoryEffect.READ, pattern=moved_tile(0))
    dst = ParamDef(kind="input", effect=MemoryEffect.WRITE, pattern=moved_tile(1))
    rmem_layout = ParamDef(kind="attribute", annotation=LayoutBase, optional=True, default=None)
    smem_layout = ParamDef(kind="attribute", annotation=LayoutBase, optional=True, default=None)

    scope = any_threads()


@register_typeinfer(Copy)
def _(call: "Call", ctx: "TypeInferContext") -> UnitType:
    return UnitType()


@register_verify_stmt(Copy)
def _(call: "Call", ctx: "VerifyContext") -> None:
    src = ctx.type_of(call.args[0])
    dst = ctx.type_of(call.args[1])
    if src.storage == dst.storage and src.shape != dst.shape:
        if not _is_copyable_shard(src, dst):
            ctx.error(call, f"Copy shape mismatch: {src.shape} vs {dst.shape}")
    if src.dtype != dst.dtype:
        ctx.error(call, f"Copy dtype mismatch: {src.dtype} vs {dst.dtype}")


def _input_params(op_type: type) -> tuple:
    return tuple(param for param in op_type._op_schema.signature if param.kind == "input")


def verify_between(call, ctx, lead: str = "") -> None:
    """Hold one call to the relations declared between its operands."""
    op_type = type(call.target)
    rules = between_rules(op_type)
    if not rules:
        return
    names = tuple(param.name for param in _input_params(op_type))
    operands = dict(zip(names, (ctx.type_of(arg) for arg in call.args)))
    for rule in rules:
        if not rule.holds(operands):
            ctx.error(call, lead + rule.refused(operands))


def verify_operands(call, ctx, label: str) -> None:
    """Hold each operand to the pattern declared for its parameter."""
    for param, arg in zip(_input_params(type(call.target)), call.args):
        if param.pattern is None:
            continue
        value = ctx.type_of(arg)
        if param.pattern.match(value) is None:
            ctx.error(
                call,
                f"{label} {param.name} is {tuple(value.shape)} "
                f"{value.dtype.name} storage={value.storage}: "
                f"{param.pattern.refusal(value)}",
            )


def _is_copyable_shard(src_ty, dst_ty) -> bool:
    """Both sides carry a ShardLayout describing the same per-thread buffer."""
    src_sl = getattr(src_ty, "layout", None)
    dst_sl = getattr(dst_ty, "layout", None)
    if not (isinstance(src_sl, ShardLayout) and isinstance(dst_sl, ShardLayout)):
        return False
    return src_sl.layout == dst_sl.layout
