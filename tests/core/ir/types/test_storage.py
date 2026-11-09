"""``resolve_storage`` rejects legacy long aliases.

Every canonical short name is resolved on the model path (each parsed tensor
annotation goes through it); a removed alias has no such witness, and accepting
one again would silently place a tensor in the wrong memory.
"""

from __future__ import annotations

import pytest

from tilefoundry.ir import expr, types


def test_legacy_long_aliases_rejected() -> None:
    for legacy in ("global", "meta"):
        with pytest.raises(ValueError, match=r"unknown storage"):
            expr.resolve_storage(legacy)


def test_surface_storage_uses_the_exact_enum_spellings() -> None:
    assert tuple(expr.resolve_storage(str(kind)) for kind in expr.StorageKind) == tuple(expr.StorageKind)
    for spelling in ("GMEM", "Gmem"):
        with pytest.raises(ValueError, match=r"unknown storage"):
            expr.resolve_storage(spelling)


def test_scalar_defaults_to_unmaterialized_storage() -> None:
    assert types.TensorType.from_scalar(expr.DType.i64).storage is expr.StorageKind.UMAT
