"""``resolve_storage`` rejects legacy long aliases.

Every canonical short name is resolved on the model path (each parsed tensor
annotation goes through it); a removed alias has no such witness, and accepting
one again would silently place a tensor in the wrong memory.
"""
from __future__ import annotations

import pytest

from tilefoundry.ir.types.storage import resolve_storage


def test_legacy_long_aliases_rejected() -> None:
    for legacy in ("global", "meta"):
        with pytest.raises(ValueError, match=r"unknown storage"):
            resolve_storage(legacy)
