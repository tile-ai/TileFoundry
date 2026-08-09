"""List available Target values and show their retained hardware documents."""

from __future__ import annotations

import sys

from tilefoundry.target import Target, registered_targets
from tilefoundry.target.hardware import format_capabilities, hardware_documents
from tilefoundry.utils.python_source import PythonExpr


def available_targets() -> tuple[Target, ...]:
    """Every value currently constructible from registered Target classes."""
    return tuple(
        target
        for _, target_type in sorted(registered_targets().items())
        for target in target_type.available()
    )


def _merged_imports(expressions: tuple[PythonExpr, ...]) -> tuple[str, ...]:
    grouped: dict[str, set[str]] = {}
    other: set[str] = set()
    for statement in (item for expression in expressions for item in expression.imports):
        prefix, separator, names = statement.partition(" import ")
        if separator and prefix.startswith("from "):
            grouped.setdefault(prefix[5:], set()).update(name.strip() for name in names.split(","))
        else:
            other.add(statement)
    imports = [
        f"from {module} import {', '.join(sorted(names))}" for module, names in grouped.items()
    ]
    return tuple(sorted((*imports, *other)))


def run_list() -> int:
    """Print every available Target as a reconstructing Python expression."""
    targets = available_targets()
    expressions = tuple(target.to_python() for target in targets)
    width = max((len(expression.text) for expression in expressions), default=0)
    lines = ["Available targets:", ""]
    lines.extend(
        f"  {expression.text:<{width}}  identity: {target.identity}"
        for target, expression in zip(targets, expressions)
    )
    lines.extend(("", *_merged_imports(expressions), ""))
    lines.extend(
        (
            "What one of them states, and where each number came from:",
            "  tilefoundry target show nvidia.h200_sxm",
        )
    )
    sys.stdout.write("\n".join(lines) + "\n")
    return 0


def target_by_identity(identity: str) -> Target:
    """Find one available Target value by its exact identity."""
    targets = available_targets()
    matches = tuple(target for target in targets if target.identity == identity)
    if len(matches) == 1:
        return matches[0]
    available = sorted(target.identity for target in targets)
    if not matches:
        raise ValueError(f"unknown target identity {identity!r}; available: {available}")
    raise ValueError(f"target identity {identity!r} is ambiguous")


def format_document_target(target: Target) -> str:
    """Format the attributed hardware documents retained by one Target."""
    return format_capabilities(hardware_documents(target))


def run_show(identity: str) -> int:
    """Show one available Target by exact identity."""
    target = target_by_identity(identity)
    has_documents = all(
        getattr(target, attribute, None) is not None
        for attribute in ("_architecture_document", "_device_document")
    )
    if has_documents:
        output = format_document_target(target)
    else:
        output = f"identity: {target.identity}\n{target.to_python().text}\nfacts: unavailable"
    sys.stdout.write(output + "\n")
    return 0


__all__ = [
    "available_targets",
    "format_document_target",
    "run_list",
    "run_show",
    "target_by_identity",
]
