#!/usr/bin/env python3
"""Reject ``typing.TYPE_CHECKING`` imports used to break cycles.

Future annotations already permit spelling-only references without a false
runtime dependency. Touched files come from pre-commit; any mention produces a
nonzero exit status.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_TYPE_CHECKING = re.compile(r"(?:^|[^\w.])(TYPE_CHECKING)\b")


_SELF = Path(__file__).resolve()


def findings(path: Path) -> list[tuple[int, str]]:
    """Every `TYPE_CHECKING` mention in *path*, as (line number, line)."""
    if path.resolve() == _SELF or path.suffix != ".py":
        return []
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
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
