"""Effect-form TIR Op ``tir.tensor.Reduce`` — axis reduction dispatched by ``ReduceKind`` tag."""

from __future__ import annotations

import isl

from tilefoundry.interfaces import access, typeinfer, verify
from tilefoundry.ir import expr, pattern, types

from .sync import WARP_SIZE

__all__ = ["Reduce", "workspace_slots"]

_ = pattern.Wildcard()
FOLDS_IN_FLOAT = (expr.DType.f32, expr.DType.f16, expr.DType.bf16)
PLAIN = pattern.Layout((pattern.Star(_),), (pattern.Star(_),))
ONE_MESH = pattern.Mesh(
    ("thread",),
    pattern.ComposedLayout(
        inner=None,
        offset=pattern.Wildcard("mesh_offset"),
        outer=pattern.Layout(
            (pattern.Star(pattern.Wildcard("mesh_shape")),),
            (pattern.Star(pattern.Wildcard("mesh_stride")),),
        ),
    ),
)


def folding(dtype: str, layout) -> pattern.Tensor:
    """A float-foldable tensor sharded over the one mesh both operands name."""
    return pattern.Tensor(
        dtype=pattern.Wildcard(dtype),
        layout=pattern.ShardLayout(layout, _, ONE_MESH),
        predicates=(pattern.In(pattern.Wildcard(dtype), FOLDS_IN_FLOAT),),
    )


def reduced_axes(op, rank: int) -> tuple[int, ...]:
    """Normalize this instruction's axes against one source rank."""
    axes = tuple(axis + rank if axis < 0 else axis for axis in op.axes)
    if any(axis < 0 or axis >= rank for axis in axes):
        raise ValueError(f"{type(op).__name__} axes {op.axes} exceed source rank {rank}")
    return axes


def result_shape(op, source) -> tuple:
    """Apply the instruction's declared keepdim policy to the source shape."""
    axes = reduced_axes(op, len(source.shape))
    return tuple(
        1 if axis in axes else extent
        for axis, extent in enumerate(source.shape)
        if op.keepdim or axis not in axes
    )


@expr.register_op(dialect="T", category="tensor")
class Reduce(expr.Op):
    """Generic axis reduction; dispatched by the ``kind`` tag."""

    capability = expr.OpCapability(None)
    execution_mesh = pattern.thread_execution_mesh()

    src = expr.ParamDef(kind="input", effect=expr.MemoryEffect.READ, pattern=folding("src_dtype", PLAIN))
    dst = expr.ParamDef(kind="input", effect=expr.MemoryEffect.WRITE, pattern=folding("dst_dtype", None))
    workspace = expr.ParamDef(
        kind="input",
        effect=expr.MemoryEffect.READ | expr.MemoryEffect.WRITE,
        pattern=pattern.tensor_in(expr.StorageKind.SMEM),
        optional=True,
        default=None,
    )
    axes = expr.ParamDef(kind="attribute", annotation=tuple)
    keepdim = expr.ParamDef(kind="attribute", annotation=bool)
    kind = expr.ParamDef(kind="attribute", annotation=expr.ReduceKind)


@typeinfer.register(Reduce)
def _(call: "Call", ctx: "TypeInferContext") -> types.UnitType:
    return types.UnitType()


def workspace_slots(source) -> int:
    """One slot per warp of the thread level ``source`` is sharded over.

    The runtime posts each warp's partial at its warp index and reads its group
    back (`cta_combine_via_workspace`); which warp takes which slot is its
    choice. A mesh naming several levels counts only its thread level: each
    CTA combines its own partials in its own shared memory.
    """
    mesh = source.layout.mesh
    names = types.topology_names(mesh)
    if "thread" in names:
        level = types.levels(mesh)[names.index("thread")]
    else:
        stated = mesh.layout.outer if isinstance(mesh.layout, types.ComposedLayout) else mesh.layout
        level = stated
    threads = 1
    for extent in types.flatten(level).shape:
        value = expr.static_dim_value(extent)
        if value is None or value <= 0:
            raise ValueError(
                f"Reduce workspace needs a static positive thread mesh, got extent {extent!r}"
            )
        threads *= value
    return -(-threads // WARP_SIZE)


@access.register_relation(Reduce)
def reduce_access(call: "Call", ctx) -> tuple[access.Relation, ...]:
    """Walk the source coordinates and collapse reduced axes into ``dst``.

    The workspace is reached at every warp slot from every source coordinate:
    which coordinate's partial lands in which slot is the runtime's choice.
    """
    op = call.target
    source = ctx.type_of(call.args[0])
    rank = len(source.shape)
    axes = reduced_axes(op, rank)
    out_shape = result_shape(op, source)
    surviving = [axis for axis in range(rank) if axis not in axes]
    came_from = {axis: axis for axis in range(rank)} if op.keepdim else dict(enumerate(surviving))
    writes_at = [
        "0" if axis not in came_from or came_from[axis] in axes else f"d{came_from[axis]}"
        for axis in range(len(out_shape))
    ]
    domain = ", ".join(f"d{axis}" for axis in range(rank))
    destination_access = access.Relation(isl.map(f"{{ [{domain}] -> [{', '.join(writes_at)}] }}"))
    inputs = [access.identity(rank), destination_access]
    if len(call.args) > 2:
        slots = workspace_slots(source)
        inputs.append(access.Relation(isl.map(f"{{ [{domain}] -> [s] : 0 <= s < {slots} }}")))
    return access.iterating(source.shape, (*inputs, destination_access))


@verify.register(Reduce)
def _(call: "Call", ctx: "verify.Context") -> None:
    if len(call.args) <= 2:
        return
    source, workspace = ctx.type_of(call.args[0]), ctx.type_of(call.args[2])
    if not (
        isinstance(source, types.TensorType)
        and isinstance(source.layout, types.ShardLayout)
        and isinstance(workspace, types.TensorType)
    ):
        return
    try:
        slots = workspace_slots(source)
    except ValueError as error:
        ctx.error(call, str(error))
    held = expr.static_dim_value(workspace.shape[0]) if len(workspace.shape) == 1 else None
    if held is None or held < slots:
        ctx.error(
            call,
            f"Reduce workspace {tuple(workspace.shape)} must be rank-1 with at least "
            f"{slots} slots, one per warp of the source mesh",
        )
