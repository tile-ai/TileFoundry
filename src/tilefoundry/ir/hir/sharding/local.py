from __future__ import annotations

from tilefoundry.evaluator.registry import register_eval
from tilefoundry.ir.core import Op
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.pattern import Tensor
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.core.registry import register_typeinfer
from tilefoundry.ir.types import TensorType
from tilefoundry.ir.types.shape_helpers import static_dim_value
from tilefoundry.ir.types.shard.shard_layout import ShardLayout, Split


@register_op
class Local(Op):
    """The current device's local view of a ``ShardLayout`` tensor."""
    x = ParamDef(kind="input", pattern=Tensor)
@register_typeinfer(Local)
def _(call: "Call", ctx: "TypeInferContext") -> TensorType:
    x_ty = ctx.type_of(call.args[0])
    if not isinstance(x_ty.layout, ShardLayout):
        ctx.error(call, "Local() input must have ShardLayout")
    # Shrink shape along Split-sharded axes by mesh-axis size.
    sl = x_ty.layout
    new_shape = list(x_ty.shape)
    for mesh_axis, attr in enumerate(sl.attrs):
        if isinstance(attr, Split):
            mesh_extent = sl.mesh.shape[mesh_axis]
            dim = new_shape[attr.axis]
            v = static_dim_value(dim)
            if v is not None:
                new_shape[attr.axis] = v // mesh_extent
            # Symbolic: leave as-is (dim.* simplification later).
    return TensorType(
        shape=tuple(new_shape),
        dtype=x_ty.dtype,
        layout=sl.layout,  # strip the shard wrapper, leaving base Layout
        storage=x_ty.storage,
    )


@register_eval(Local)
def _eval_local(ctx):
    # Single modelled participant: the local view is the operand itself.
    return ctx.args[0]
