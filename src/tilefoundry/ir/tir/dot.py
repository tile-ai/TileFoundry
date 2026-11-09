"""Effect-form TIR Op ``tir.tensor.Dot`` — ``dst = sum(lhs * rhs)`` in one statement.

See [tir §2.3](docs/spec/tir.md#23-tir-ops).
"""

from __future__ import annotations

import isl

from tilefoundry.interfaces import access, typeinfer, verify
from tilefoundry.ir import dim, expr, pattern, types, utils

from .reduce import workspace_slots

__all__ = ["Dot"]


@expr.register_op(dialect="T", category="tensor")
class Dot(expr.Op):
    """Fused multiply-contract; every participant leaves holding the total."""

    execution_mesh = pattern.thread_execution_mesh()

    lhs = expr.ParamDef(kind="input", effect=expr.MemoryEffect.READ, pattern=pattern.is_ranked_tensor())
    rhs = expr.ParamDef(kind="input", effect=expr.MemoryEffect.READ, pattern=pattern.is_ranked_tensor())
    dst = expr.ParamDef(kind="input", effect=expr.MemoryEffect.WRITE, pattern=pattern.is_ranked_tensor())
    workspace = expr.ParamDef(
        kind="input",
        effect=expr.MemoryEffect.READ | expr.MemoryEffect.WRITE,
        pattern=pattern.tensor_in(expr.StorageKind.SMEM),
        optional=True,
        default=None,
    )


@typeinfer.register(Dot)
def _(call: "Call", ctx: "TypeInferContext") -> types.UnitType:
    return types.UnitType()


def local_numel(ty) -> int | None:
    """The elements one participant holds of *ty*, or ``None`` when undecidable."""
    layout = getattr(ty, "layout", None)
    if isinstance(layout, types.ShardLayout):
        try:
            shape = types.shard_layout_local_shape(layout, require_static=False)
        except ValueError:
            return None
    else:
        shape = ty.shape
    n = 1
    for dim_value in shape:
        if not isinstance(dim_value, int):
            return None
        n *= dim_value
    return n


@verify.register(Dot)
def _(call: "Call", ctx: "verify.Context") -> None:
    lhs = ctx.type_of(call.args[0])
    rhs = ctx.type_of(call.args[1])
    dst = ctx.type_of(call.args[2])

    lhs_n = local_numel(lhs)
    rhs_n = local_numel(rhs)
    if lhs_n is not None and rhs_n is not None and lhs_n != rhs_n:
        ctx.error(
            call,
            f"Dot operands contract over different lengths: {lhs_n} vs {rhs_n}",
        )
    dst_n = local_numel(dst)
    if dst_n is not None and dst_n != 1:
        ctx.error(call, f"Dot destination must be one cell, got {dst_n} elements")

    if len(call.args) < 4:
        return
    if not isinstance(getattr(lhs, "layout", None), types.ShardLayout):
        ctx.error(
            call,
            "Dot with a workspace contracts across the block, and the warps "
            "that post a partial are the ones lhs's mesh names; lhs must "
            "carry a ShardLayout",
        )


@access.register_relation(Dot)
def dot_access(call, ctx):
    """Pair inputs by their per-thread register index, then collapse the total."""
    type_list = tuple(ctx.type_of(arg) for arg in call.args)
    length = local_numel(type_list[0])
    if length is None:
        raise ValueError("Dot access needs a static local register count")
    values = {}
    unit = types.Mesh((types.Topology("thread", 1),), types.Layout((1,), (1,)), ("unit",))

    def registers(type_):
        layout = type_.layout
        shard = (
            layout if isinstance(layout, types.ShardLayout) else types.ShardLayout(layout, (types.Broadcast(),), unit)
        )
        coordinates = dim.layout_to_isl_map(
            type_.shape, shard, values, divided=lambda layer: range(len(layer.attrs))
        )
        strides = utils.register_strides(shard)
        modes = ", ".join(f"p{axis}" for axis in range(len(strides)))
        index = " + ".join(f"{step} * p{axis}" for axis, step in enumerate(strides) if step) or "0"
        slots = coordinates.apply_range(isl.map(f"{{ [{modes}] -> [{index}] }}"))
        return slots.intersect_domain(dim.shape_to_isl_set(type_.shape, values)).reverse()

    reads = [registers(type_) for type_ in type_list[:2]]
    result = registers(type_list[2])
    destination = isl.map("{ [r] -> [0] }").apply_range(result)
    boundaries = [*reads, destination]
    if len(type_list) > 3:
        slots = workspace_slots(type_list[0]) if isinstance(type_list[0].layout, types.ShardLayout) else 1
        boundaries.append(isl.map(f"{{ [r] -> [s] : 0 <= s < {slots} }}"))
    boundaries.append(destination)

    def acc(relation):
        names = {
            relation.get_dim_name(isl.dim_type.PARAM, i)
            for i in range(relation.dim(isl.dim_type.PARAM))
        }
        return access.Relation(relation, {name: values[name] for name in names})

    return access.iterating((max(1, length),), tuple(acc(relation) for relation in boundaries))
