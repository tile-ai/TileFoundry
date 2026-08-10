"""Where the files that ship with TileFoundry are read from.

Each kind is either in the checkout being worked in or in the installation's data
prefix, and the same two-step lookup answers for all of them: one kind per thing
that ships, rather than one copy of the lookup per thing.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, distribution
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
    "orchestrator": Kind(source=("tests", "models", "orchestrator"), installed="orchestrator"),
    "tutorial": Kind(source=("docs", "tutorial"), installed="tutorial"),
}


def _distribution_directory(known: Kind) -> Path | None:
    """Find *known* through the installed distribution's data-file records."""
    try:
        installed = distribution("tilefoundry")
    except PackageNotFoundError:
        return None

    records = installed.files or ()
    prefix = ("share", "tilefoundry", known.installed)
    for record in records:
        parts = record.parts
        for start in range(len(parts) - len(prefix) + 1):
            if parts[start : start + len(prefix)] != prefix:
                continue
            found = Path(installed.locate_file(record))
            for _ in parts[start + len(prefix) :]:
                found = found.parent
            if found.is_dir():
                return found
    return None


def directory(kind: str) -> Path:
    """The directory *kind* is read from: this checkout when it is one, else the installation.

    The directory *kind* is read from: this checkout when it is one, else the
    installation. The checkout comes first because it is the only place the two
    can disagree, and there the working copy is what its author means.
    """
    known = _KINDS[kind]
    source = _REPOSITORY_ROOT.joinpath(*known.source)
    if source.is_dir():
        return source


    from sysconfig import get_path  # noqa: PLC0415

    installed = Path(get_path("data")) / "share" / "tilefoundry" / known.installed
    if installed.is_dir():
        return installed
    installed = _distribution_directory(known)
    if installed is not None:
        return installed
    raise FileNotFoundError(f"installed TileFoundry {kind} directory was not found")


def path(kind: str, name: str) -> Path:
    """One shipped file by name, from wherever that kind is read from."""
    found = directory(kind) / name
    if not found.is_file():
        raise FileNotFoundError(f"installed TileFoundry {name} was not found")
    return found


def _manifest_files(destination: str) -> tuple[Path, ...]:
    """The checkout files named by one data-files destination, sorted by name."""
    with (_REPOSITORY_ROOT / "pyproject.toml").open("rb") as manifest:
        data_files = tomllib.load(manifest)["tool"]["setuptools"]["data-files"]
    try:
        return tuple(
            sorted(
                (_REPOSITORY_ROOT / entry for entry in data_files[destination]),
                key=lambda path: path.name,
            )
        )
    except KeyError:
        raise FileNotFoundError(
            f"installed TileFoundry data directory {destination} was not found"
        ) from None


def directories(kind: str) -> tuple[Path, ...]:
    """The shipped directories of *kind*, in the checkout or an installation."""
    known = _KINDS[kind]
    root = directory(kind)
    source = _REPOSITORY_ROOT.joinpath(*known.source)
    if root != source:
        return tuple(sorted(path for path in root.iterdir() if path.is_dir()))

    prefix = f"share/tilefoundry/{known.installed}/"
    with (_REPOSITORY_ROOT / "pyproject.toml").open("rb") as manifest:
        data_files = tomllib.load(manifest)["tool"]["setuptools"]["data-files"]
    names = {
        Path(destination.removeprefix(prefix)).parts[0]
        for destination in data_files
        if destination.startswith(prefix)
    }
    return tuple(root / name for name in sorted(names))


def files(kind: str, name: str) -> tuple[Path, ...]:
    """The files a shipped *kind* directory carries, sorted by filename."""
    known = _KINDS[kind]
    root = directory(kind)
    found = root / name
    if root == _REPOSITORY_ROOT.joinpath(*known.source):
        destination = f"share/tilefoundry/{known.installed}/{name}"
        return _manifest_files(destination)
    if not found.is_dir():
        raise FileNotFoundError(f"installed TileFoundry {kind} {name} was not found")
    return tuple(sorted((entry for entry in found.iterdir() if entry.is_file()), key=lambda path: path.name))


def model_files(name: str) -> tuple[Path, ...]:
    """The files a shipped model directory carries, sorted by filename."""
    return files("models", name)


__all__ = ["Kind", "directories", "directory", "files", "model_files", "path"]
