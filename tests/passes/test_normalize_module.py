"""``normalize_to_module()`` boundary + ``CompilerOptions`` canonical text."""

from __future__ import annotations

import pytest

from tilefoundry import CompilerOptions, normalize_to_module
from tilefoundry.ir.core.module import Module


def test_normalize_rejects_invalid_inputs() -> None:
    """Non-IR / missing entry raises early at the module-first boundary."""
    with pytest.raises(TypeError, match="normalize_to_module"):
        normalize_to_module(42)
    with pytest.raises(ValueError):
        normalize_to_module(Module(name="bad", functions=(), entry="nonexistent"))


def test_compiler_options_canonical_text_per_target() -> None:
    """Default target is ``cuda``; canonical text is deterministic + target-sensitive."""
    default = CompilerOptions()
    assert default.target == "cuda"
    assert "target=cuda" in default.canonical_text()

    cuda1 = CompilerOptions(target="cuda")
    cuda2 = CompilerOptions(target="cuda")
    hip = CompilerOptions(target="hip")
    assert cuda1.canonical_text() == cuda2.canonical_text()
    assert cuda1.canonical_text() != hip.canonical_text()
