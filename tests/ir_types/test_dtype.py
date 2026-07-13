"""DType declarations: the low-precision members and their canonical spellings."""
from __future__ import annotations

from tilefoundry.ir.types import DType


def test_low_precision_dtypes_declared():
    for name in ("fp8e4m3", "f8e8m0", "f4e2m1"):
        assert name in DType.__members__, f"missing DType {name}"
        assert DType[name].value == name
    # fp8e4m3 is the sole canonical fp8 spelling; no alternate is introduced.
    assert "f8e4m3" not in DType.__members__
