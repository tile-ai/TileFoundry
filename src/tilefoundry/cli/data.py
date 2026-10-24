"""Where the files that ship with TileFoundry are read from.

Reference pages live in the checkout or installation data prefix. Models and
orchestrators are package resources in both environments.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, distribution, entry_points
from importlib.resources import files as resource_files
from pathlib import Path
from typing import Any

from tilefoundry.target import load_targets

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class Kind:
    """One kind of shipped file: where it lives here, and where it lands installed."""

    source: tuple[str, ...]
    installed: str


_KINDS = {
    "spec": Kind(source=("docs", "spec"), installed="spec"),
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
    """Read reference pages from checkout or installation, and core package resources.

    The checkout comes first because it is the only place the two can disagree,
    and there the working copy is what its author means. Orchestrators always
    come from core package resources.
    """
    if kind == "orchestrator":
        return Path(resource_files("tilefoundry") / "orchestrator")
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
        raise FileNotFoundError(f"TileFoundry {kind} file {name} was not found")
    return found


def directories(kind: str) -> tuple[Path, ...]:
    """The shipped directories of *kind*, excluding interpreter cache files."""
    return tuple(
        sorted(
            path
            for path in directory(kind).iterdir()
            if path.is_dir() and path.name != "__pycache__"
        )
    )


def files(kind: str, name: str) -> tuple[Path, ...]:
    """The files a shipped *kind* directory carries, sorted by filename."""
    root = directory(kind)
    found = root / name
    if not found.is_dir():
        names = ", ".join(path.name for path in directories(kind))
        raise FileNotFoundError(f"no {kind} named {name!r}; there are {names}")
    return tuple(
        sorted((entry for entry in found.iterdir() if entry.is_file()), key=lambda path: path.name)
    )


def model_catalog() -> dict[str, tuple[dict[str, Any], Path]]:
    """Index the models shipped by loaded targets, rejecting ambiguous names."""
    loaded = load_targets()
    models = {}
    owners = {}
    for ep in sorted(entry_points(group="tilefoundry.targets"), key=lambda ep: ep.name):
        if ep.name not in loaded:
            continue
        root = Path(resource_files(ep.value) / "models")
        if not root.is_dir():
            continue
        catalog = json.loads((root / "catalog.json").read_text(encoding="utf-8"))
        for model in catalog["models"]:
            name = model["name"]
            if name in models:
                raise ValueError(
                    f"model {name!r} is shipped by both targets {owners[name]!r} and {ep.name!r}"
                )
            models[name] = model, root / name
            owners[name] = ep.name
    return dict(sorted(models.items()))


def model_files(name: str) -> tuple[Path, ...]:
    """The files a shipped model directory carries, sorted by filename."""
    root = model_catalog()[name][1]
    return tuple(sorted(path for path in root.iterdir() if path.is_file()))


__all__ = [
    "Kind",
    "directories",
    "directory",
    "files",
    "model_catalog",
    "model_files",
    "path",
]
