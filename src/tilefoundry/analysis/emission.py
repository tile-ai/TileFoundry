"""What lowering emits for an HIR call, asked by lowering and by the analyses that follow it.

Which Calls are dimensions, which instruction a call issues and with which
operand types, and which mesh a copy or an instruction is owned by. The rules
are those of [passes §7.1](docs/spec/passes.md#71-converthirtotir).
"""

from __future__ import annotations

from dataclasses import replace

from tilefoundry.ir.core import Call, Op
from tilefoundry.ir.hir.math.binary import Binary as HirBinary
from tilefoundry.ir.hir.schedule import ScheduleOp
from tilefoundry.ir.hir.sharding.mesh_coord import MeshCoord
from tilefoundry.ir.types import Broadcast, Mesh, Partial, ShardLayout
from tilefoundry.ir.types.dim import is_dim_op_call
from tilefoundry.ir.types.utils import participant_layout
from tilefoundry.visitor_registry.candidates import candidate_operands, sole_candidate

from .access import AliasKind, BufferAlias


def lowers_to_dim(call: Call) -> bool:
    """Whether *call* computes a dimension: a dim op, a mesh coordinate or scalar arithmetic."""
    target = call.target
    return (
        is_dim_op_call(call)
        or isinstance(target, MeshCoord)
        or (isinstance(target, HirBinary) and call.type.shape == ())
    )


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


def _participants(layouts) -> Mesh | None:
    """The participant mesh of the first ``ShardLayout`` among *layouts*, if any."""
    return next(
        (participant_layout(layout).mesh for layout in layouts if isinstance(layout, ShardLayout)),
        None,
    )


def copy_mesh(types) -> Mesh | None:
    """The mesh a copy between operands of *types* runs on, None for the enclosing one."""
    return _participants(type_.layout for type_ in types)


def issue_owner(types) -> Mesh | None:
    """The mesh that owns an instruction issued with operands of *types*, if one does.

    ``Partial`` attributes count as ``Broadcast`` here.
    """
    return _participants(
        replace(
            type_.layout,
            attrs=tuple(
                Broadcast() if isinstance(attr, Partial) else attr for attr in type_.layout.attrs
            ),
        )
        if isinstance(type_.layout, ShardLayout)
        else type_.layout
        for type_ in types
    )


def accepted_aliases(value: "AllocationValue") -> tuple[BufferAlias, ...]:
    """Accepted links that lowering executes; a register IN_PLACE tie shares no storage."""
    return tuple(
        alias
        for alias in value.aliases
        if value.lifetime.memory_level != "rmem" or alias.kind is not AliasKind.IN_PLACE
    )


__all__ = [
    "accepted_aliases",
    "copy_mesh",
    "issue_owner",
    "lowers_to_dim",
    "operand_types",
    "selected_op",
]
