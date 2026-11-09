"""Provide parser-owned ``Tensor`` and ``ConstTensor`` annotation sugar.

Both resolve to ``TensorType``; ``ConstTensor`` additionally marks the parsed
parameter as an external constant.

See [parser §2.1](docs/spec/parser.md#21-syntax).
"""

from __future__ import annotations

from tilefoundry.ir import expr, types


def tensor_type_getitem(args) -> types.TensorType:
    if not isinstance(args, tuple):
        args = (args,)
    shape = args[0]
    dtype_val = args[1] if len(args) > 1 else expr.DType.f32
    if isinstance(dtype_val, str):
        dtype_val = expr.DType.from_name(dtype_val)
    if not isinstance(shape, tuple):
        shape = (shape,)
    layout = None
    storage = expr.StorageKind.GMEM
    if len(args) > 2:
        layout_or_storage = args[2]
        if isinstance(layout_or_storage, (str, expr.StorageKind)):
            storage = layout_or_storage
        else:
            layout = layout_or_storage
    if len(args) > 3 and args[3]:
        storage = args[3]
    return types.TensorType(
        shape=shape,
        dtype=dtype_val,
        layout=types.Layout(shape) if layout is None else layout,
        storage=expr.resolve_storage(storage),
    )


class Tensor:
    """Subscriptable annotation surface for the DSL.

    ``Tensor[(M, K), f32]`` → ``TensorType(shape=(M, K), dtype=f32)``.
    Used by the ``@func`` parser to resolve type annotations.
    """

    def __class_getitem__(cls, args):
        return tensor_type_getitem(args)


class ConstTensor:
    """Subscriptable annotation surface for an external constant tensor.

    ``ConstTensor[(M, K), f32]`` resolves to the identical ``TensorType`` as
    ``Tensor[(M, K), f32]``; only the parsed parameter ``Var.is_const`` flag
    differs.
    """

    def __class_getitem__(cls, args):
        return tensor_type_getitem(args)


__all__ = ["ConstTensor", "Tensor"]
