#!/usr/bin/env python3
"""Reject `typing.TYPE_CHECKING` as a way to break an import cycle.

A type-only cycle has two shapes. One puts the import behind
``if TYPE_CHECKING:`` and quotes the annotation; the other just quotes the
annotation and does not import at all. Both type-check. Only the second is
honest about what the module needs at runtime: the first states an import that
never happens, so a reader cannot tell a real dependency from a spelling aid,
and any tool that walks imports sees an edge that is not there.

`from __future__ import annotations` is already on in this project, so every
annotation is a string and the quoted form costs nothing.

Ruff's own TC001-TC003 push the opposite way -- they move runtime imports *into*
a `TYPE_CHECKING` block -- so this rule cannot be delegated to it.

Run over the files a commit touches (the pre-commit hook passes them); the exit
status is non-zero if any was found.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

#: Both the import of the flag and any use of it. Importing it and never
#: branching on it is dead weight; branching on it is the shim itself.
_TYPE_CHECKING = re.compile(r"(?:^|[^\w.])(TYPE_CHECKING)\b")

#: This file states the pattern, so its own source is not scanned for it.
_SELF = Path(__file__).resolve()


def findings(path: Path) -> list[tuple[int, str]]:
    """Every `TYPE_CHECKING` mention in *path*, as (line number, line)."""
    if path.resolve() == _SELF or path.suffix != ".py":
        return []
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []  # unreadable or binary: nothing to claim about it
    return [
        (number, line.strip())
        for number, line in enumerate(text.splitlines(), start=1)
        if _TYPE_CHECKING.search(line)
    ]


def main(argv: list[str]) -> int:
    if not argv:
        print(
            "usage: forward_references_lint.py <path> ...  "
            "(the pre-commit hook passes the staged files)",
            file=sys.stderr,
        )
        return 2
    failed = False
    for name in argv:
        path = Path(name)
        for number, line in findings(path):
            failed = True
            print(f"{path}:{number}: `TYPE_CHECKING` shim")
            print(f"    {line}")
    if failed:
        print(
            "\nA type-only cycle MUST be broken by quoting the annotation and "
            "dropping the import, not by hiding the import behind "
            "`if TYPE_CHECKING:`. The module's import list is then what it "
            "actually needs to run.",
            file=sys.stderr,
        )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
