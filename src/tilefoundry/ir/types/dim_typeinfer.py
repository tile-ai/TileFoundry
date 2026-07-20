from __future__ import annotations

from ..core.registry import register_typeinfer
from .dim import (
    DimAdd,
    DimConst,
    DimFloorDiv,
    DimMax,
    DimMin,
    DimMod,
    DimMul,
    DimSub,
    DimVar,
)
from .tensor_type import TensorType


def _meta_i64() -> TensorType:
    return TensorType.meta_scalar()


for _cls in (DimConst, DimVar, DimAdd, DimSub, DimMul, DimFloorDiv, DimMod, DimMin, DimMax):

    @register_typeinfer(_cls)
    def _(call, ctx, _cls=_cls):  # noqa: ARG001 — uniform signature
        return _meta_i64()
