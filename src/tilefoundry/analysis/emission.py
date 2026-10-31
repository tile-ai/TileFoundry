"""What lowering emits for each HIR event, decided in one place.

Lowering branches on these answers, and anything that needs to know which
statements lowering emits, which instruction a call issues and which operands it
reads and writes asks the same code. The rules are those of
[passes §7.1](docs/spec/passes.md#71-converthirtotir).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from enum import Enum, auto

from tilefoundry.ir.core import Call, Op, SSAValue
from tilefoundry.ir.core.param_def import MemoryEffect, ParamDef
from tilefoundry.ir.hir.loop_region import LoopRegion
from tilefoundry.ir.hir.math.binary import Binary as HirBinary
from tilefoundry.ir.hir.schedule import ScheduleOp
from tilefoundry.ir.hir.sharding.mesh_coord import MeshCoord
from tilefoundry.ir.types import StorageKind, TensorType
from tilefoundry.ir.types.dim import is_dim_op_call
from tilefoundry.visitor_registry.buffer_alias import aliased_operand
from tilefoundry.visitor_registry.candidates import candidate_operands, sole_candidate

from .access import AliasKind, BufferAlias

type AliasesOf = Callable[[SSAValue], Sequence[BufferAlias]]

_VIEWS = ("TupleGetItem", "Slice", "Reshape", "Bitcast")


class Emission(Enum):
    """The statements lowering emits for one HIR call.

    ``DIMENSION``, ``VIEW`` and ``ALLOCATE`` emit none; ``CALL`` calls a retained
    function; ``FILL`` emits one ``Fill``; ``INSERT_SLICE`` emits the copies
    :func:`copies_insert_slice_base` and :func:`copies_insert_slice_update` name;
    ``INSTRUCTION`` issues :func:`selected_op`; lowering refuses ``UNSCHEDULED``.
    """

    DIMENSION = auto()
    CALL = auto()
    VIEW = auto()
    ALLOCATE = auto()
    FILL = auto()
    UNSCHEDULED = auto()
    INSERT_SLICE = auto()
    INSTRUCTION = auto()


def emission(call: Call, *, retained: bool) -> Emission:
    """What lowering emits for *call*; *retained* says it calls a retained function."""
    target = call.target
    if (
        is_dim_op_call(call)
        or isinstance(target, MeshCoord)
        or (isinstance(target, HirBinary) and call.type.shape == ())
    ):
        return Emission.DIMENSION
    if retained:
        return Emission.CALL
    name = type(target).__name__
    if name in _VIEWS or (name == "Reshard" and aliased_operand(call) is not None):
        return Emission.VIEW
    if name == "Empty":
        return Emission.ALLOCATE
    if name in ("Zeros", "FullLike"):
        return Emission.FILL
    if name == "IndexSelect":
        return Emission.UNSCHEDULED
    if name == "InsertSlice":
        return Emission.INSERT_SLICE
    return Emission.INSTRUCTION


def selected_op(call: Call, target) -> Op | None:
    """The instruction *call* issues on *target*: its schedule's, else the sole candidate."""
    if isinstance(call.target, ScheduleOp):
        return call.target.op
    return sole_candidate(call, target=target)


def operand_types(call: Call, op: Op) -> tuple:
    """The types *op* is issued with, one per parameter it takes for *call*."""
    if isinstance(call.target, ScheduleOp):
        return tuple(arg.type for arg in call.args)
    return candidate_operands(call, type(op))[1]


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


def required_execution_mesh(op: Op):
    """The execution mesh *op* declares, or None when lowering issues it once.

    With a declared mesh, lowering issues *op* once for every point
    ``IssuePlan.points`` enumerates; without one, once in its owner region.
    """
    declaration = getattr(op, "atom", op)
    if isinstance(declaration, type):
        return declaration.execution_mesh
    return getattr(declaration, "required_execution_mesh", None)


def staged_scalar(type_) -> bool:
    """Whether lowering first fills an operand of *type_* into a register scalar."""
    return (
        isinstance(type_, TensorType) and type_.shape == () and type_.storage is StorageKind.UMAT
    )


def accepted_aliases(value: "AllocationValue") -> tuple[BufferAlias, ...]:
    """Accepted links that lowering executes; a register IN_PLACE tie shares no storage."""
    return tuple(
        alias
        for alias in value.aliases
        if value.lifetime.memory_level != "rmem" or alias.kind is not AliasKind.IN_PLACE
    )


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
    "Emission",
    "accepted_aliases",
    "copies_carry_backedge",
    "copies_carry_entry",
    "copies_insert_slice_base",
    "copies_insert_slice_update",
    "emission",
    "operand_roles",
    "operand_types",
    "required_execution_mesh",
    "selected_op",
    "staged_scalar",
]
