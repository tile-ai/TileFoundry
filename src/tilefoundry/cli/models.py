"""The `models` command: which models are described, and what one of them looks like."""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from typing import Any

from tilefoundry.cli import data


def catalog() -> dict[str, Any]:
    """The shipped catalog. The installed package carries no corpus, so the
    inventory is read from data rather than built by importing anything."""
    return json.loads(data.path("models", "catalog.json").read_text(encoding="utf-8"))


def _find(name: str) -> dict[str, Any]:
    known = catalog()["models"]
    for model in known:
        if model["name"] == name:
            return model
    names = ", ".join(model["name"] for model in known)
    raise ValueError(f"no model named {name!r}; the catalog has {names}")


def _counts(model: dict[str, Any]) -> str:
    counts = model["counts"]
    return f"{counts['leaf_modules']} leaf modules, {counts['functions']} functions"


def render_models() -> str:
    """The inventory, oracles first and everything else marked as not one."""
    known = catalog()
    oracle = known["oracle_level"]
    models = known["models"]
    width = max(len(model["name"]) for model in models)

    lines = [f"Models in {data.directory('models')}:", ""]
    for heading, chosen in (
        (f"Verified at {oracle}, usable as an oracle:",
         [m for m in models if m["level"] == oracle]),
        ("Below that, and so not usable as an oracle:",
         [m for m in models if m["level"] != oracle]),
    ):
        lines.append(heading)
        if not chosen:
            lines += ["  none", ""]
            continue
        for model in chosen:
            lines.append(f"  {model['level']}  {model['name']:<{width}}  {_counts(model)}")
        lines.append("")
    lines.append("Levels:")
    for level, meaning in sorted(known["levels"].items()):
        lines.append(f"  {level}  {meaning}")
    return "\n".join(lines) + "\n"


def _label(node: dict[str, Any]) -> str:
    """A node's Modules by name: one, or a run written as the range it covers."""
    names = node["names"]
    if len(names) == 1:
        return names[0]
    return f"{names[0]}..{names[-1]}  ({len(names)} identical, each as shown)"


def _tree(node: dict[str, Any], depth: int) -> list[str]:
    indent = "  " * depth
    mark = "*" if node["leaf"] else " "
    lines = [f"{mark} {indent}{_label(node)}"]
    lines += [f"    {indent}{signature}" for signature in node["functions"]]
    for child in node["modules"]:
        lines += _tree(child, depth + 1)
    return lines


def render_model(name: str) -> str:
    """One model's whole forest, every root it publishes.

    `*` marks a leaf Module, one with no child Modules. A run of adjacent,
    identically shaped, consecutively numbered Modules is one entry naming its range
    and how many it stands for.
    """
    model = _find(name)
    lines = [
        f"{model['name']}  ({model['level']}: {model['evidence']})",
        _counts(model),
        "",
    ]
    for root in model["modules"]:
        lines += _tree(root, 0)
    return "\n".join(lines) + "\n"


def source_summary(path: Path) -> str:
    """The first docstring line in *path*, or a marker when it has none."""
    try:
        source = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return "-"
    docstring = ast.get_docstring(source)
    return docstring.splitlines()[0] if docstring else "-"


def render_source_directory(directory: Path, files: tuple[Path, ...]) -> str:
    """One source directory followed by aligned leading file descriptions."""
    width = max(len(path.name) for path in files)
    lines = [str(directory)]
    lines += [f"{path.name:<{width}}  {source_summary(path)}" for path in files]
    return "\n".join(lines) + "\n"


def model_source(name: str) -> str:
    """The shipped model directory followed by one source summary per file."""
    _find(name)
    files = data.model_files(name)
    directory = data.directory("models") / name
    return render_source_directory(directory, files)


def run_models(name: str | None, *, source: bool = False) -> int:
    """Print the inventory, one model's forest, or one shipped model directory."""
    if name is None:
        if source:
            raise ValueError("--source needs a model to print the source of")
        sys.stdout.write(render_models())
    elif source:
        sys.stdout.write(model_source(name))
    else:
        sys.stdout.write(render_model(name))
    return 0


__all__ = [
    "catalog",
    "model_source",
    "render_model",
    "render_models",
    "render_source_directory",
    "run_models",
    "source_summary",
]
