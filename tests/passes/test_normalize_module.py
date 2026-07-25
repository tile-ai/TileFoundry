"""``CompilerOptions`` canonical text."""

from __future__ import annotations

from tilefoundry import CompilerOptions


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
