"""Effect-form TIR Op ``tir.tensor.Reduce`` — axis reduction dispatched by ``ReduceKind`` tag."""

from __future__ import annotations

import isl

from tilefoundry.ir.core import Op, OpCapability
from tilefoundry.ir.core.kinds import ReduceKind
from tilefoundry.ir.core.param_def import MemoryEffect, ParamDef
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.pattern import (
    LayoutPattern,
    MeshPattern,
    ShardLayoutPattern,
    StarPattern,
    TensorPattern,
    WildcardPattern,
    utils,
)
from tilefoundry.ir.pattern import predicates as P
from tilefoundry.ir.types import DType, StorageKind, UnitType
from tilefoundry.visitor_registry import register_typeinfer, register_verify_stmt
from tilefoundry.visitor_registry.access_relation import (
    AccessRelations,
    AffineAccess,
    BoundaryRelation,
    identity_access,
    iterating,
    register_access_relation,
)

__all__ = ["ReduceKind", "Reduce"]

_ = WildcardPattern()
_FOLDS_IN_FLOAT = (DType.f32, DType.f16, DType.bf16)
_PLAIN = LayoutPattern((StarPattern(_),), (StarPattern(_),))
_ONE_MESH = MeshPattern(
    ("thread",),
    LayoutPattern(
        (StarPattern(WildcardPattern("mesh_shape")),),
        (StarPattern(WildcardPattern("mesh_stride")),),
    ),
)


def _folding(dtype: str, layout) -> TensorPattern:
    """A float-foldable tensor sharded over the one mesh both operands name."""
    return TensorPattern(
        dtype=WildcardPattern(dtype),
        layout=ShardLayoutPattern(layout, _, _ONE_MESH),
        predicates=(P.In(WildcardPattern(dtype), _FOLDS_IN_FLOAT),),
    )


@register_op(dialect="T", category="tensor")
class Reduce(Op):
    """Generic axis reduction; dispatched by the ``kind`` tag."""

    capability = OpCapability(None)
    execution_mesh = utils.thread_execution_mesh()

    src = ParamDef(kind="input", effect=MemoryEffect.READ, pattern=_folding("src_dtype", _PLAIN))
    dst = ParamDef(kind="input", effect=MemoryEffect.WRITE, pattern=_folding("dst_dtype", None))
    workspace = ParamDef(
        kind="input",
        effect=MemoryEffect.READ | MemoryEffect.WRITE,
        pattern=utils.tensor_in(StorageKind.SMEM),
        optional=True,
        default=None,
    )
    axes = ParamDef(kind="attribute", annotation=tuple)
    keepdim = ParamDef(kind="attribute", annotation=bool)
    kind = ParamDef(kind="attribute", annotation=ReduceKind)


@register_typeinfer(Reduce)
def _(call: "Call", ctx: "TypeInferContext") -> UnitType:
    return UnitType()


@register_access_relation(Reduce)
def _reduce_access(call: "Call", ctx) -> AccessRelations:
    """Walk the source coordinates and collapse reduced axes into ``dst``."""
    source = ctx.type_of(call.args[0])
    destination = ctx.type_of(call.args[1])
    rank = len(source.shape)
    axes = tuple(axis + rank if axis < 0 else axis for axis in call.target.axes)
    surviving = [axis for axis in range(rank) if axis not in axes]
    came_from = (
        {axis: axis for axis in range(rank)}
        if len(destination.shape) == rank
        else dict(enumerate(surviving))
    )
    writes_at = [
        "0" if axis not in came_from or came_from[axis] in axes else f"d{came_from[axis]}"
        for axis in range(len(destination.shape))
    ]
    domain = ", ".join(f"d{axis}" for axis in range(rank))
    destination_access = BoundaryRelation(
        AffineAccess(isl.map(f"{{ [{domain}] -> [{', '.join(writes_at)}] }}"))
    )
    inputs = [BoundaryRelation(identity_access(rank)), destination_access]
    if len(call.args) > 2:
        workspace = ctx.type_of(call.args[2])
        workspace_rank = len(workspace.shape)
        workspace_coords = ", ".join(
            f"d{axis}" if axis < rank else "0" for axis in range(workspace_rank)
        )
        inputs.append(
            BoundaryRelation(
                AffineAccess(isl.map(f"{{ [{domain}] -> [{workspace_coords}] }}"))
            )
        )
    return iterating(
        source.shape,
        AccessRelations(inputs=tuple(inputs), outputs=(destination_access,)),
    )


@register_verify_stmt(Reduce)
def _(call: "Call", ctx: "VerifyContext") -> None:
    op = call.target
    if not isinstance(op.kind, ReduceKind):
        ctx.error(call, f"Reduce: kind must be ReduceKind enum, got {type(op.kind)}")
    src_ty = ctx.type_of(call.args[0])  # noqa: F841
    dst_ty = ctx.type_of(call.args[1])  # noqa: F841
