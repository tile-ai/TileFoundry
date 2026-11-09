"""TIR view Expr Op: `tir.memory.TensorView`.

Constructs a logical tensor view over a typed pointer. ``layout`` can be a plain ``Layout``
or a ``ShardLayout`` (→ shard tensor view, no allocation).
"""

from __future__ import annotations

from tilefoundry.interfaces import typeinfer
from tilefoundry.ir import expr, types

from .ptr_of import PtrOf


@expr.register_op(name="tensor_view", dialect='T', category='memory')
class TensorView(expr.Op):
    """Derive a sub-view of a tensor (value form).

    ``layout`` updates the ``ShardLayout`` / cute ``Layout``; the optional
    ``shape`` overrides the logical shape. When omitted, it is inherited only
    from a syntactic ``PtrOf(tensor)`` input. Integer inputs are byte offsets
    from the dynamic shared-memory base and require dtype, storage, and shape.
    """

    pointer = expr.ParamDef(kind="input")
    dtype = expr.ParamDef(kind="attribute", annotation=str, optional=True, default=None)
    storage = expr.ParamDef(kind="attribute", annotation=expr.StorageKind, optional=True, default=None)
    layout = expr.ParamDef(kind="attribute", annotation=types.LayoutBase)
    shape = expr.ParamDef(kind="attribute", annotation=tuple, default=None)


@typeinfer.register(TensorView)
def _(call: "expr.Call", ctx: "TypeInferContext") -> types.TensorType:
    if len(call.args) > 1:
        ctx.error(call, "coordinates are not supported on a pointer view")
    pointer = ctx.type_of(call.args[0])
    op = call.target
    origin = call.args[0]
    inherited_shape = None
    if isinstance(origin, expr.Call):
        if isinstance(origin.target, PtrOf):
            tensor = ctx.type_of(origin.args[0])
            if not isinstance(tensor, types.TensorType):
                ctx.error(call, "PtrOf input must have TensorType")
            inherited_shape = tensor.shape

    if not isinstance(pointer, types.PointerType):
        address = call.args[0]
        if (
            not isinstance(address, expr.Constant)
            or isinstance(address.value, bool)
            or not isinstance(address.value, int)
        ):
            ctx.error(call, "tensor_view input must be a pointer")
        if op.dtype is None or op.storage is None or op.shape is None:
            ctx.error(
                call,
                "shared-memory byte offset requires dtype, storage, and shape",
            )
        try:
            storage = expr.resolve_storage(op.storage)
            pointer = types.PointerType(expr.DType.from_name(op.dtype), storage)
        except (TypeError, ValueError) as error:
            ctx.error(call, str(error))
        if pointer.storage is not expr.StorageKind.SMEM:
            ctx.error(call, "integer tensor_view address requires storage='smem'")
        address.retype(pointer)
    elif op.dtype is not None or op.storage is not None:
        try:
            stated = types.PointerType(
                pointer.dtype if op.dtype is None else expr.DType.from_name(op.dtype),
                pointer.storage if op.storage is None else expr.resolve_storage(op.storage),
            )
        except (TypeError, ValueError) as error:
            ctx.error(call, str(error))
        if stated != pointer:
            ctx.error(call, "tensor_view pointer dtype/storage mismatch")

    new_shape = op.shape if op.shape is not None else inherited_shape
    if new_shape is None:
        ctx.error(call, "tensor_view shape is required unless pointer is T.ptr_of(tensor)")
    return types.TensorType(
        shape=new_shape, dtype=pointer.dtype, layout=op.layout, storage=pointer.storage
    )


__all__ = ["TensorView"]
