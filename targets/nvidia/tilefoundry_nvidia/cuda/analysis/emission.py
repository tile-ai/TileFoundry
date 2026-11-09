"""What CUDA lowering emits for a Call, which the ``sync`` family reads too.

The operands an instruction reads and fills, the mesh it is issued on, and the
copies lowering adds where memory accepted no alias. See
[analysis §1](targets/nvidia/tilefoundry_nvidia/cuda/docs/spec/analysis.md#1-sync).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from tilefoundry import analysis
from tilefoundry.ir import core, expr, hir, types, utils
from tilefoundry_nvidia.cuda.ir.tir import memory

type AliasesOf = Callable[[core.SSAValue], Sequence[analysis.BufferAlias]]


def operand_roles(
    call: expr.Call, params: Sequence[expr.ParamDef]
) -> tuple[tuple[expr.ParamDef, ...], tuple[expr.ParamDef, ...]]:
    """The parameters *call*'s arguments supply, in order, and those its result fills.

    A scheduled call supplies every parameter, its destination included; an
    automatically selected one supplies the parameters it reads.
    """
    reads = tuple(param for param in params if param.effect & expr.MemoryEffect.READ)
    writes = tuple(param for param in params if param.effect & expr.MemoryEffect.WRITE)
    produced = tuple(param for param in writes if not param.effect & expr.MemoryEffect.READ)
    return (tuple(params) if isinstance(call.target, hir.schedule.ScheduleOp) else reads), produced


def result_param(params: Sequence[expr.ParamDef]) -> expr.ParamDef:
    """The parameter an instruction's result is; one it also reads is written in place."""
    return next(param for param in params if param.effect & expr.MemoryEffect.WRITE)


def required_execution_mesh(op: expr.Op):
    """The execution mesh *op* declares, or None when lowering issues it once.

    With a declared mesh, lowering issues *op* once for every point
    ``IssuePlan.points`` enumerates; without one, once in its owner region.
    """
    declaration = getattr(op, "atom", op)
    if isinstance(declaration, type):
        return declaration.execution_mesh
    return getattr(declaration, "required_execution_mesh", None)


def issue_mesh(op: expr.Op, owner: types.Mesh | None, enclosing: types.Mesh | None) -> types.Mesh | None:
    """The mesh lowering issues *op* on, None for the enclosing one.

    An instruction without an execution mesh is issued on its owner; one with
    an execution mesh on the participant frame of that mesh within the owner, or
    within *enclosing* when nothing owns it.
    """
    required = required_execution_mesh(op)
    if required is None:
        return owner
    return utils.participant_frame(owner or enclosing, required)[0]


def tied(aliases: Sequence[analysis.BufferAlias], kind: analysis.AliasKind, target: core.SSAValue) -> bool:
    return any(alias.kind is kind and alias.target == target for alias in aliases)


def copies_carry_entry(loop: core.LoopRegion, slot: int, aliases_of: AliasesOf) -> bool:
    """Whether a carried slot's initial value is copied into separate phi storage."""
    return not tied(
        aliases_of(core.SSAValue(loop.params[slot])), analysis.AliasKind.CARRY_INIT, core.SSAValue(loop.args[slot])
    )


def copies_carry_backedge(loop: core.LoopRegion, slot: int, aliases_of: AliasesOf) -> bool:
    """Whether a carried slot's yielded value is copied into its phi storage."""
    return not tied(
        aliases_of(core.SSAValue(loop.yield_values[slot])),
        analysis.AliasKind.CARRY_YIELD,
        core.SSAValue(loop.params[slot]),
    )


def copies_insert_slice_base(call: expr.Call, aliases_of: AliasesOf) -> bool:
    """Whether an InsertSlice copies its destination into the result first."""
    return not tied(aliases_of(core.SSAValue(call)), analysis.AliasKind.INSERT_SLICE, core.SSAValue(call.args[0]))


def copies_insert_slice_update(call: expr.Call, aliases_of: AliasesOf) -> bool:
    """Whether an InsertSlice copies its update into the result window."""
    return not tied(aliases_of(core.SSAValue(call.args[1])), analysis.AliasKind.WINDOW, core.SSAValue(call))


def tensor_map_load(op: expr.Op, operand_types: Sequence[types.TensorType]) -> bool:
    """Whether *op*, issued with *operand_types*, loads a tensor-map box into shared memory.

    Lowering issues such a load on the mbarrier words the ``sync`` plan places.
    """
    return (
        isinstance(op, memory.CopyAsyncTensor)
        and operand_types[1].storage is expr.StorageKind.SMEM
    )


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
    "tensor_map_load",
]
