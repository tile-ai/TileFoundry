"""TIR view Expr Op: `tir.memory.TensorView`.

Constructs a logical tensor view over a typed pointer. ``layout`` can be a plain ``Layout``
or a ``ShardLayout`` (→ shard tensor view, no allocation).

A slice view carries an absolute element-start coordinate per axis after the
memory source: a single coordinate is a flat rank-1 window, while multiple
coordinates are a per-axis N-D window.
"""

from __future__ import annotations

from tilefoundry.ir.core import Call, Constant, Op
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.types import DType, PointerType, StorageKind, TensorType
from tilefoundry.ir.types.layout import Layout, LayoutBase
from tilefoundry.ir.types.storage import resolve_storage
from tilefoundry.ir.types.stride import compact_row_major
from tilefoundry.visitor_registry import register_typeinfer


@register_op(name="tensor_view")
class TensorView(Op):
    """Derive a sub-view of a tensor (value form).

    ``layout`` updates the ``ShardLayout`` / cute ``Layout``; the optional
    ``shape`` overrides the logical shape. When omitted, it is inherited only
    from a syntactic ``PtrOf(tensor)`` input. Integer inputs are byte offsets
    from the dynamic shared-memory base and require dtype, storage, and shape.
    """

    pointer = ParamDef(kind="input")
    dtype = ParamDef(kind="attribute", annotation=str, optional=True, default=None)
    storage = ParamDef(kind="attribute", annotation=StorageKind, optional=True, default=None)
    layout = ParamDef(kind="attribute", annotation=LayoutBase)
    shape = ParamDef(kind="attribute", annotation=tuple, default=None)


@register_typeinfer(TensorView)
def _(call: "Call", ctx: "TypeInferContext") -> TensorType:
    pointer = ctx.type_of(call.args[0])
    op = call.target
    origin = call.args[0]
    inherited_shape = None
    if isinstance(origin, Call):
        from .ptr_of import PtrOf  # noqa: PLC0415

        if isinstance(origin.target, PtrOf):
            tensor = ctx.type_of(origin.args[0])
            if not isinstance(tensor, TensorType):
                ctx.error(call, "PtrOf input must have TensorType")
            inherited_shape = tensor.shape

    if not isinstance(pointer, PointerType):
        address = call.args[0]
        if (
            not isinstance(address, Constant)
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
            storage = resolve_storage(op.storage)
            pointer = PointerType(DType.from_name(op.dtype), storage)
        except (TypeError, ValueError) as error:
            ctx.error(call, str(error))
        if pointer.storage is not StorageKind.SMEM:
            ctx.error(call, "integer tensor_view address requires storage='smem'")
        address.type = pointer
    elif op.dtype is not None or op.storage is not None:
        try:
            stated = PointerType(
                pointer.dtype if op.dtype is None else DType.from_name(op.dtype),
                pointer.storage if op.storage is None else resolve_storage(op.storage),
            )
        except (TypeError, ValueError) as error:
            ctx.error(call, str(error))
        if stated != pointer:
            ctx.error(call, "tensor_view pointer dtype/storage mismatch")

    new_shape = op.shape if op.shape is not None else inherited_shape
    if new_shape is None:
        ctx.error(call, "tensor_view shape is required unless pointer is T.ptr_of(tensor)")
    return TensorType(
        shape=new_shape,
        dtype=pointer.dtype,
        layout=op.layout,
        storage=pointer.storage,
    )


def _c_order_strides(src_shape: tuple) -> list:
    """C-order contiguous strides of the source buffer a slice view reads."""
    return list(compact_row_major(tuple(src_shape)))


def layout_for_slice(src_shape: tuple, axis: int, sliced_shape: tuple) -> Layout:
    """Compute a plain Layout for a slice view."""
    view_strides = _c_order_strides(src_shape)
    view_strides.pop(axis)
    return Layout(shape=sliced_shape, strides=tuple(view_strides))


TensorView.layout_for_slice = staticmethod(layout_for_slice)


def layout_for_slice_nd(src_shape: tuple, sliced_shape: tuple) -> Layout:
    """Plain Layout for an N-D window: the sub-block keeps the source's C-order strides.

    Plain Layout for an N-D window: the sub-block keeps the source's C-order
    strides, so it is a strided view of the original buffer (all axes retained,
    each shrunk to the window extent).
    """
    return Layout(shape=sliced_shape, strides=tuple(_c_order_strides(src_shape)))


TensorView.layout_for_slice_nd = staticmethod(layout_for_slice_nd)
