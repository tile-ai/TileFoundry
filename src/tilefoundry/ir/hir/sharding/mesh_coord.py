"""The coordinate of the unit a region is running on."""

from __future__ import annotations

from tilefoundry.evaluator.registry import register_eval
from tilefoundry.evaluator.value import EvalError
from tilefoundry.ir.core import Op
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.mesh_scope import covered_by_scope
from tilefoundry.ir.pattern import Scalar
from tilefoundry.ir.types import DType, TensorType
from tilefoundry.ir.types.layout import flatten
from tilefoundry.ir.types.mesh import Mesh
from tilefoundry.ir.types.storage import StorageKind
from tilefoundry.ir.types.utils import static_dim_value
from tilefoundry.visitor_registry import register_typeinfer
from tilefoundry.visitor_registry.access_relation import (
    measures_without_reading,
    register_access_relation,
)
from tilefoundry.visitor_registry.contexts import TypeInferResults


@register_op
class MeshCoord(Op):
    """This unit's coordinate along one axis of *mesh*.

    Naming a mesh axis and asking which unit this is are two different
    questions. The first fills in a ``Split`` on a layout and leaves no node
    behind; only the second needs an operation, and this is it. So the mesh is a
    value the node carries rather than the scope it sits in, and whether an
    enclosing region binds that mesh stays a rule about the regions. The axis is
    an operand rather than a name so it can be computed, and so the node reads
    like every other Op in this dialect.
    """

    mesh = ParamDef(kind="attribute", annotation=Mesh)
    axis = ParamDef(kind="input", pattern=Scalar)


@register_typeinfer(MeshCoord)
def _(call: "Call", ctx: "TypeInferContext") -> TypeInferResults:
    """A coordinate is one number about this unit, so it carries no placement."""
    if not isinstance(call.target.mesh, Mesh):
        ctx.error(call, "MeshCoord.mesh must be a Mesh")
    if ctx.current_mesh is None or not covered_by_scope(call.target.mesh, ctx.current_mesh):
        ctx.error(call, "MeshCoord.mesh must be bound by the current mesh scope")
    if not call.args:
        ctx.error(call, "missing required input 'axis'")
    shape = flatten(call.target.mesh.layout).shape
    axis = static_dim_value(call.args[0])
    if axis is not None and not 0 <= axis < len(shape):
        ctx.error(call, f"axis {axis} is out of range for rank-{len(shape)} mesh")
    extents = (shape[axis],) if axis is not None else shape
    bounds = (
        (0, max(extents))
        if extents
        and all(isinstance(extent, int) and not isinstance(extent, bool) for extent in extents)
        else None
    )
    return TypeInferResults(
        TensorType(shape=(), dtype=DType.i64, layout=None, storage=StorageKind.RMEM),
        value_range=bounds,
    )


@register_eval(MeshCoord)
def _eval_mesh_coord(ctx):
    raise EvalError(
        "MeshCoord is not modelled: evaluation runs one mesh participant, so there "
        "is no unit for a coordinate to be the coordinate of "
        "(docs/spec/evaluator.md section 6)."
    )


register_access_relation(MeshCoord)(measures_without_reading)


__all__ = ["MeshCoord"]
