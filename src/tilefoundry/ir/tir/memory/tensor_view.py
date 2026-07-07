"""TIR view Expr Op: `tir.memory.TensorView`.

Constructs a logical tensor view over a memory source
(tensor / ptr / span). ``layout`` can be a plain ``Layout`` (→ plain view)
or a ``ShardLayout`` (→ shard tensor view, no allocation).

When called with a second arg (index Var), emits a row-slice view:
``cute::local_tile(x, make_shape(K), make_coord(index))``.
"""

from __future__ import annotations

from tilefoundry.ir.core import Op
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.pattern import Tensor
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.core.registry import register_typeinfer
from tilefoundry.ir.types import TensorType
from tilefoundry.ir.types.shard.layout import Layout


@register_op(name="tensor_view")
class TensorView(Op):
    """Derive a sub-view of a tensor (value form).

    ``layout`` updates the ``ShardLayout`` / cute ``Layout``; the optional
    ``shape`` overrides the logical shape (reshape). ``memory`` MAY be a
    ``PtrOf`` result (ptr + offset).
    """
    memory = ParamDef(kind="input", pattern=Tensor)
    layout = ParamDef(kind="attribute", annotation=object)
    shape = ParamDef(kind="attribute", annotation=tuple, default=None)

@register_typeinfer(TensorView)
def _(call: "Call", ctx: "TypeInferContext") -> TensorType:
    src_ty = ctx.type_of(call.args[0])
    op = call.target
    new_layout = op.layout
    new_shape = op.shape if op.shape is not None else src_ty.shape
    # Slice: second arg is the index Var, shape from call.type
    if len(call.args) > 1:
        new_shape = call.type.shape
    return TensorType(
        shape=new_shape,
        dtype=src_ty.dtype,
        layout=new_layout,
        storage=src_ty.storage,
    )

def layout_for_slice(src_shape: tuple, axis: int, sliced_shape: tuple) -> Layout:
    """Compute a plain Layout for a slice view."""
    rank = len(src_shape)  # noqa: F841
    strides = [1]
    for s in reversed(src_shape[1:]):
        strides.insert(0, strides[0] * s)
    view_strides = list(strides)
    view_strides.pop(axis)
    return Layout(shape=sliced_shape, strides=tuple(view_strides))

TensorView.layout_for_slice = staticmethod(layout_for_slice)
