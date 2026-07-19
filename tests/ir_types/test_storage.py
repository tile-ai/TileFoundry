"""``resolve_storage`` accepts canonical short names + ``StorageKind`` / ``None``
and rejects legacy long aliases."""
from __future__ import annotations

import pytest

from tilefoundry.ir.types.storage import StorageKind, resolve_storage


def test_canonical_short_names_resolve() -> None:
    expected = (
        ("host", StorageKind.HOST),
        ("gmem", StorageKind.GMEM),
        ("smem", StorageKind.SMEM),
        ("rmem", StorageKind.RMEM),
        ("tmem", StorageKind.TMEM),
    )
    for name, kind in expected:
        assert resolve_storage(name) is kind
        assert str(kind) == name


def test_storagekind_and_none_pass_through() -> None:
    assert resolve_storage(StorageKind.GMEM) is StorageKind.GMEM
    assert resolve_storage(None) is None


def test_legacy_long_aliases_rejected() -> None:
    for legacy in ("global", "shared", "reg", "register", "meta"):
        with pytest.raises(ValueError, match=r"unknown storage"):
            resolve_storage(legacy)
