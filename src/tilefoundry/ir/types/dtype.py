from __future__ import annotations

import enum


class DType(enum.Enum):
    f32 = "f32"
    f16 = "f16"
    bf16 = "bf16"
    fp8e4m3 = "fp8e4m3"
    f8e8m0 = "f8e8m0"
    f4e2m1 = "f4e2m1"
    i32 = "i32"
    i64 = "i64"
    bool = "bool"


# Cast-boundary dtypes: values live only through Cast, and generic arithmetic
# (Binary / Unary / MatMul / Reduce) rejects them at type inference.
LOW_PRECISION_DTYPES: frozenset[DType] = frozenset(
    {DType.fp8e4m3, DType.f8e8m0, DType.f4e2m1}
)


__all__ = ["DType", "LOW_PRECISION_DTYPES"]
