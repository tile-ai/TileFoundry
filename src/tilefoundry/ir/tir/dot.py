"""Effect-form TIR Op ``tir.tensor.Dot`` — ``dst = sum(lhs * rhs)`` in one statement.

See [tir §2.3](docs/spec/tir.md#23-tir-ops).
"""

from __future__ import annotations

import isl

from tilefoundry.ir.core import Op
from tilefoundry.ir.core.param_def import MemoryEffect, ParamDef
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.isl_interop import layout_to_isl_map, shape_to_isl_set
from tilefoundry.ir.pattern import is_ranked_tensor, utils
from tilefoundry.ir.tir.reduce import _workspace_slots
from tilefoundry.ir.types import Layout, Mesh, StorageKind, Topology, UnitType
from tilefoundry.ir.types.shard_layout import (
    Broadcast,
    ShardLayout,
    register_strides,
    shard_layout_local_shape,
)
from tilefoundry.visitor_registry import register_typeinfer, register_verify_stmt
from tilefoundry.visitor_registry.access_relation import (
    AccessRelation,
    iterating,
    register_access_relation,
)

__all__ = ["Dot"]


@register_op(dialect="T", category="tensor")
class Dot(Op):
    """Fused multiply-contract; every participant leaves holding the total."""

    execution_mesh = utils.thread_execution_mesh()

    lhs = ParamDef(kind="input", effect=MemoryEffect.READ, pattern=is_ranked_tensor())
    rhs = ParamDef(kind="input", effect=MemoryEffect.READ, pattern=is_ranked_tensor())
    dst = ParamDef(kind="input", effect=MemoryEffect.WRITE, pattern=is_ranked_tensor())
    workspace = ParamDef(
        kind="input",
        effect=MemoryEffect.READ | MemoryEffect.WRITE,
        pattern=utils.tensor_in(StorageKind.SMEM),
        optional=True,
        default=None,
    )


@register_typeinfer(Dot)
def _(call: "Call", ctx: "TypeInferContext") -> UnitType:
    return UnitType()


def _local_numel(ty) -> int | None:
    """The elements one participant holds of *ty*, or ``None`` when undecidable."""
    layout = getattr(ty, "layout", None)
    if isinstance(layout, ShardLayout):
        try:
            shape = shard_layout_local_shape(layout, require_static=False)
        except ValueError:
            return None
    else:
        shape = ty.shape
    n = 1
    for dim in shape:
        if not isinstance(dim, int):
            return None
        n *= dim
    return n


@register_verify_stmt(Dot)
def _(call: "Call", ctx: "VerifyContext") -> None:
    lhs = ctx.type_of(call.args[0])
    rhs = ctx.type_of(call.args[1])
    dst = ctx.type_of(call.args[2])

    lhs_n = _local_numel(lhs)
    rhs_n = _local_numel(rhs)
    if lhs_n is not None and rhs_n is not None and lhs_n != rhs_n:
        ctx.error(
            call,
            f"Dot operands contract over different lengths: {lhs_n} vs {rhs_n}",
        )
    dst_n = _local_numel(dst)
    if dst_n is not None and dst_n != 1:
        ctx.error(call, f"Dot destination must be one cell, got {dst_n} elements")

    if len(call.args) < 4:
        return
    if not isinstance(getattr(lhs, "layout", None), ShardLayout):
        ctx.error(
            call,
            "Dot with a workspace contracts across the block, and the warps "
            "that post a partial are the ones lhs's mesh names; lhs must "
            "carry a ShardLayout",
        )


@register_access_relation(Dot)
def _dot_access(call, ctx):
    """Pair inputs by their per-thread register index, then collapse the total."""
    types = tuple(ctx.type_of(arg) for arg in call.args)
    length = _local_numel(types[0])
    if length is None:
        raise ValueError("Dot access needs a static local register count")
    values = {}
    unit = Mesh((Topology("thread", 1),), Layout((1,), (1,)), ("unit",))

    def registers(type_):
        layout = type_.layout
        shard = (
            layout if isinstance(layout, ShardLayout) else ShardLayout(layout, (Broadcast(),), unit)
        )
        coordinates = layout_to_isl_map(
            type_.shape, shard, values, divided=lambda layer: range(len(layer.attrs))
        )
        strides = register_strides(shard)
        modes = ", ".join(f"p{axis}" for axis in range(len(strides)))
        index = " + ".join(f"{step} * p{axis}" for axis, step in enumerate(strides) if step) or "0"
        slots = coordinates.apply_range(isl.map(f"{{ [{modes}] -> [{index}] }}"))
        return slots.intersect_domain(shape_to_isl_set(type_.shape, values)).reverse()

    reads = [registers(type_) for type_ in types[:2]]
    result = registers(types[2])
    destination = isl.map("{ [r] -> [0] }").apply_range(result)
    boundaries = [*reads, destination]
    if len(types) > 3:
        slots = _workspace_slots(types[0]) if isinstance(types[0].layout, ShardLayout) else 1
        boundaries.append(isl.map(f"{{ [r] -> [s] : 0 <= s < {slots} }}"))
    boundaries.append(destination)

    def access(relation):
        names = {
            relation.get_dim_name(isl.dim_type.PARAM, i)
            for i in range(relation.dim(isl.dim_type.PARAM))
        }
        return AccessRelation(relation, {name: values[name] for name in names})

    return iterating((max(1, length),), tuple(access(relation) for relation in boundaries))
