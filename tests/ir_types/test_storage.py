"""``resolve_storage`` accepts canonical short names + ``StorageKind`` / ``None``
and rejects legacy long aliases."""
from __future__ import annotations

import pytest

from tilefoundry.ir.types.storage import StorageKind, resolve_storage


def test_canonical_short_names_resolve() -> None:
    assert resolve_storage("host") is StorageKind.HOST
    assert resolve_storage("gmem") is StorageKind.GMEM
    assert resolve_storage("smem") is StorageKind.SMEM
    assert resolve_storage("rmem") is StorageKind.RMEM
    assert resolve_storage("tmem") is StorageKind.TMEM


def test_storagekind_and_none_pass_through() -> None:
    assert resolve_storage(StorageKind.GMEM) is StorageKind.GMEM
    assert resolve_storage(None) is None


def test_legacy_long_aliases_rejected() -> None:
    for legacy in ("global", "shared", "reg", "register", "meta"):
        with pytest.raises(ValueError, match=r"unknown storage"):
            resolve_storage(legacy)
