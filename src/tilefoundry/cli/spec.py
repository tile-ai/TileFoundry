"""Where an installed specification document is read from."""

from __future__ import annotations

from pathlib import Path

_HELP_SPEC_TOPICS = {
    "cli": "cli",
    "dsl": "hir",
}


def _source_spec_path(topic: str) -> Path:
    """Find one source-tree spec used by editable and direct invocations."""
    spec_name = _HELP_SPEC_TOPICS.get(topic, topic)
    return Path(__file__).resolve().parents[3] / "docs" / "spec" / f"{spec_name}.md"


def spec_path(topic: str) -> Path:
    """Return an installed spec path, falling back to the source tree."""
    spec_name = _HELP_SPEC_TOPICS.get(topic, topic)
    source_path = _source_spec_path(topic)
    if source_path.is_file():
        return source_path

    # setuptools data-files are placed below Python's installation data prefix.
    from sysconfig import get_path  # noqa: PLC0415

    installed = (
        Path(get_path("data"))
        / "share"
        / "tilefoundry"
        / "spec"
        / f"{spec_name}.md"
    )
    if installed.is_file():
        return installed
    raise FileNotFoundError(f"installed TileFoundry {spec_name} spec was not found")


def dsl_spec_path() -> Path:
    """Return the HIR spec exposed by the historical ``dsl`` help topic."""
    return spec_path("dsl")


def read_spec(topic: str) -> str:
    """Read the single source of truth for a `tilefoundry help` topic."""
    return spec_path(topic).read_text(encoding="utf-8")


def read_dsl_spec() -> str:
    """Read the HIR spec exposed by the historical ``dsl`` help topic."""
    return read_spec("dsl")


__all__ = ["dsl_spec_path", "read_dsl_spec", "read_spec", "spec_path"]
