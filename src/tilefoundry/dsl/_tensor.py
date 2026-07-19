"""Authoring-surface ``Tensor`` / ``ConstTensor`` annotation sugar.

``Tensor[(M, K), "f32"]`` is the DSL annotation surface that the
``@func`` parser uses to resolve type annotations on `@func` /
`@prim_func` parameters and return types. It is a parser-owned
authoring sugar, not part of the IR type system; the IR type carrier
is `tilefoundry.ir.types.TensorType`.

``ConstTensor[(M, K), "f32"]`` resolves to the same ``TensorType`` as
``Tensor[(M, K), "f32"]``; it marks the parsed parameter ``Var`` with
``is_const=True`` (external constant tensor, e.g. a resident weight).
"""

from __future__ import annotations

from tilefoundry.ir.types import DType, TensorType
from tilefoundry.ir.types.storage import StorageKind, resolve_storage


def _tensor_type_getitem(args) -> TensorType:
    if not isinstance(args, tuple):
        args = (args,)
    shape = args[0]
    dtype_val = args[1] if len(args) > 1 else DType.f32
    if isinstance(dtype_val, str):
        member = getattr(DType, dtype_val, None)
        if not isinstance(member, DType):
            raise ValueError(f"DType: unknown value {dtype_val!r}")
        dtype_val = member
    if not isinstance(shape, tuple):
        shape = (shape,)
    layout = args[2] if len(args) > 2 else None
    storage = args[3] if len(args) > 3 and args[3] else StorageKind.GMEM
    return TensorType(
        shape=shape, dtype=dtype_val, layout=layout,
        storage=resolve_storage(storage),
    )


class Tensor:
    """Subscriptable annotation surface for the DSL.

    ``Tensor[(M, K), f32]`` → ``TensorType(shape=(M, K), dtype=f32)``.
    Used by the ``@func`` parser to resolve type annotations.
    """

    def __class_getitem__(cls, args):
        return _tensor_type_getitem(args)


class ConstTensor:
    """Subscriptable annotation surface for an external constant tensor.

    ``ConstTensor[(M, K), f32]`` resolves to the identical ``TensorType`` as
    ``Tensor[(M, K), f32]``; only the parsed parameter ``Var.is_const`` flag
    differs.
    """

    def __class_getitem__(cls, args):
        return _tensor_type_getitem(args)


__all__ = ["ConstTensor", "Tensor"]
