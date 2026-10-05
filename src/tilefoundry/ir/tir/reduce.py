"""Effect-form TIR Op ``tir.tensor.Reduce`` — axis reduction dispatched by ``ReduceKind`` tag."""

from __future__ import annotations

import isl

from tilefoundry.ir.core import Op, OpCapability
from tilefoundry.ir.core.kinds import ReduceKind
from tilefoundry.ir.core.param_def import MemoryEffect, ParamDef
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.pattern import (
    ComposedLayoutPattern,
    LayoutPattern,
    MeshPattern,
    ShardLayoutPattern,
    StarPattern,
    TensorPattern,
    WildcardPattern,
    utils,
)
from tilefoundry.ir.pattern import predicates as P
from tilefoundry.ir.tir.sync import WARP_SIZE
from tilefoundry.ir.types import DType, ShardLayout, StorageKind, UnitType
from tilefoundry.ir.types.layout import flatten
from tilefoundry.ir.types.utils import static_dim_value
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
    ComposedLayoutPattern(
        inner=None,
        offset=WildcardPattern("mesh_offset"),
        outer=LayoutPattern(
            (StarPattern(WildcardPattern("mesh_shape")),),
            (StarPattern(WildcardPattern("mesh_stride")),),
        ),
    ),
)


def _folding(dtype: str, layout) -> TensorPattern:
    """A float-foldable tensor sharded over the one mesh both operands name."""
    return TensorPattern(
        dtype=WildcardPattern(dtype),
        layout=ShardLayoutPattern(layout, _, _ONE_MESH),
        predicates=(P.In(WildcardPattern(dtype), _FOLDS_IN_FLOAT),),
    )


def _reduced_axes(op, rank: int) -> tuple[int, ...]:
    """Normalize this instruction's axes against one source rank."""
    axes = tuple(axis + rank if axis < 0 else axis for axis in op.axes)
    if any(axis < 0 or axis >= rank for axis in axes):
        raise ValueError(f"{type(op).__name__} axes {op.axes} exceed source rank {rank}")
    return axes


def _result_shape(op, source) -> tuple:
    """Apply the instruction's declared keepdim policy to the source shape."""
    axes = _reduced_axes(op, len(source.shape))
    return tuple(
        1 if axis in axes else extent
        for axis, extent in enumerate(source.shape)
        if op.keepdim or axis not in axes
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


def _workspace_slots(source) -> int:
    """One slot per warp of the thread mesh ``source`` is sharded over.

    The runtime posts each warp's partial at its warp index and reads its group
    back (`cta_combine_via_workspace`); which warp takes which slot is its choice.
    """
    threads = 1
    for extent in flatten(source.layout.mesh.layout).shape:
        value = static_dim_value(extent)
        if value is None or value <= 0:
            raise ValueError(
                f"Reduce workspace needs a static positive thread mesh, got extent {extent!r}"
            )
        threads *= value
    return -(-threads // WARP_SIZE)


@register_access_relation(Reduce)
def _reduce_access(call: "Call", ctx) -> AccessRelations:
    """Walk the source coordinates and collapse reduced axes into ``dst``.

    The workspace is reached at every warp slot from every source coordinate:
    which coordinate's partial lands in which slot is the runtime's choice.
    """
    op = call.target
    source = ctx.type_of(call.args[0])
    rank = len(source.shape)
    axes = _reduced_axes(op, rank)
    out_shape = _result_shape(op, source)
    surviving = [axis for axis in range(rank) if axis not in axes]
    came_from = (
        {axis: axis for axis in range(rank)}
        if op.keepdim
        else dict(enumerate(surviving))
    )
    writes_at = [
        "0" if axis not in came_from or came_from[axis] in axes else f"d{came_from[axis]}"
        for axis in range(len(out_shape))
    ]
    domain = ", ".join(f"d{axis}" for axis in range(rank))
    destination_access = BoundaryRelation(
        AffineAccess(isl.map(f"{{ [{domain}] -> [{', '.join(writes_at)}] }}"))
    )
    inputs = [BoundaryRelation(identity_access(rank)), destination_access]
    if len(call.args) > 2:
        slots = _workspace_slots(source)
        inputs.append(
            BoundaryRelation(AffineAccess(isl.map(f"{{ [{domain}] -> [s] : 0 <= s < {slots} }}")))
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
    source = ctx.type_of(call.args[0])
    if len(call.args) <= 2 or not isinstance(source.layout, ShardLayout):
        return
    workspace = ctx.type_of(call.args[2])
    try:
        slots = _workspace_slots(source)
    except ValueError as error:
        ctx.error(call, str(error))
    held = static_dim_value(workspace.shape[0]) if len(workspace.shape) == 1 else None
    if held is None or held < slots:
        ctx.error(
            call,
            f"Reduce workspace {tuple(workspace.shape)} must be rank-1 with at least "
            f"{slots} slots, one per warp of the source mesh",
        )
