"""Where the files that ship with TileFoundry are read from.

Each kind is either in the checkout being worked in or in the installation's data
prefix, and the same two-step lookup answers for all of them: one kind per thing
that ships, rather than one copy of the lookup per thing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class Kind:
    """One kind of shipped file: where it lives here, and where it lands installed."""

    source: tuple[str, ...]
    installed: str


_KINDS = {
    "spec": Kind(source=("docs", "spec"), installed="spec"),
    "models": Kind(source=("tests", "models"), installed="models"),
    "tutorial": Kind(source=("docs", "tutorial"), installed="tutorial"),
}


def directory(kind: str) -> Path:
    """The directory *kind* is read from: this checkout when it is one, else the
    installation. The checkout comes first because it is the only place the two
    can disagree, and there the working copy is what its author means."""
    known = _KINDS[kind]
    source = _REPOSITORY_ROOT.joinpath(*known.source)
    if source.is_dir():
        return source

    # setuptools data-files are placed below Python's installation data prefix.
    from sysconfig import get_path  # noqa: PLC0415

    installed = Path(get_path("data")) / "share" / "tilefoundry" / known.installed
    if installed.is_dir():
        return installed
    raise FileNotFoundError(f"installed TileFoundry {kind} directory was not found")


def path(kind: str, name: str) -> Path:
    """One shipped file by name, from wherever that kind is read from."""
    found = directory(kind) / name
    if not found.is_file():
        raise FileNotFoundError(f"installed TileFoundry {name} was not found")
    return found


__all__ = ["Kind", "directory", "path"]
