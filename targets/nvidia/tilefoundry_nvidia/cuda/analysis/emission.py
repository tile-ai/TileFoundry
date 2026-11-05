"""What CUDA lowering emits for a Call, which the ``sync`` family reads too.

The operands an instruction reads and fills, the mesh it is issued on, and the
copies lowering adds where memory accepted no alias. See
[analysis §1](targets/nvidia/tilefoundry_nvidia/cuda/docs/spec/analysis.md#1-sync).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from tilefoundry.analysis.access import AliasKind, BufferAlias
from tilefoundry.ir.core import Call, Op, SSAValue
from tilefoundry.ir.core.param_def import MemoryEffect, ParamDef
from tilefoundry.ir.hir.loop_region import LoopRegion
from tilefoundry.ir.hir.schedule import ScheduleOp
from tilefoundry.ir.types import Mesh
from tilefoundry.ir.types.utils import _participant_frame

type AliasesOf = Callable[[SSAValue], Sequence[BufferAlias]]


def operand_roles(
    call: Call, params: Sequence[ParamDef]
) -> tuple[tuple[ParamDef, ...], tuple[ParamDef, ...]]:
    """The parameters *call*'s arguments supply, in order, and those its result fills.

    A scheduled call supplies every parameter, its destination included; an
    automatically selected one supplies the parameters it reads.
    """
    reads = tuple(param for param in params if param.effect & MemoryEffect.READ)
    writes = tuple(param for param in params if param.effect & MemoryEffect.WRITE)
    produced = tuple(param for param in writes if not param.effect & MemoryEffect.READ)
    return (tuple(params) if isinstance(call.target, ScheduleOp) else reads), produced


def result_param(params: Sequence[ParamDef]) -> ParamDef:
    """The parameter an instruction's result is; one it also reads is written in place."""
    return next(param for param in params if param.effect & MemoryEffect.WRITE)


def required_execution_mesh(op: Op):
    """The execution mesh *op* declares, or None when lowering issues it once.

    With a declared mesh, lowering issues *op* once for every point
    ``IssuePlan.points`` enumerates; without one, once in its owner region.
    """
    declaration = getattr(op, "atom", op)
    if isinstance(declaration, type):
        return declaration.execution_mesh
    return getattr(declaration, "required_execution_mesh", None)


def issue_mesh(op: Op, owner: Mesh | None, enclosing: Mesh | None) -> Mesh | None:
    """The mesh lowering issues *op* on, None for the enclosing one.

    An instruction without an execution mesh is issued on its owner; one with
    an execution mesh on the participant frame of that mesh within the owner, or
    within *enclosing* when nothing owns it.
    """
    required = required_execution_mesh(op)
    if required is None:
        return owner
    return _participant_frame(owner or enclosing, required)[0]


def _tied(aliases: Sequence[BufferAlias], kind: AliasKind, target: SSAValue) -> bool:
    return any(alias.kind is kind and alias.target == target for alias in aliases)


def copies_carry_entry(loop: LoopRegion, slot: int, aliases_of: AliasesOf) -> bool:
    """Whether a carried slot's initial value is copied into separate phi storage."""
    return not _tied(
        aliases_of(SSAValue(loop.params[slot])), AliasKind.CARRY_INIT, SSAValue(loop.args[slot])
    )


def copies_carry_backedge(loop: LoopRegion, slot: int, aliases_of: AliasesOf) -> bool:
    """Whether a carried slot's yielded value is copied into its phi storage."""
    return not _tied(
        aliases_of(SSAValue(loop.yield_values[slot])),
        AliasKind.CARRY_YIELD,
        SSAValue(loop.params[slot]),
    )


def copies_insert_slice_base(call: Call, aliases_of: AliasesOf) -> bool:
    """Whether an InsertSlice copies its destination into the result first."""
    return not _tied(aliases_of(SSAValue(call)), AliasKind.INSERT_SLICE, SSAValue(call.args[0]))


def copies_insert_slice_update(call: Call, aliases_of: AliasesOf) -> bool:
    """Whether an InsertSlice copies its update into the result window."""
    return not _tied(aliases_of(SSAValue(call.args[1])), AliasKind.WINDOW, SSAValue(call))


__all__ = [
    "AliasesOf",
    "copies_carry_backedge",
    "copies_carry_entry",
    "copies_insert_slice_base",
    "copies_insert_slice_update",
    "issue_mesh",
    "operand_roles",
    "required_execution_mesh",
    "result_param",
]
