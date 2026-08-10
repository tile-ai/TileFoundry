#!/usr/bin/env python3
"""Reject committed CJK, Hangul, Cyrillic, Arabic, or Hebrew script.

Latin accents and Greek mathematical notation remain valid. The pre-commit hook
passes touched files; any rejected character produces a nonzero exit status.
"""

from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path

_NON_LATIN = re.compile("[　-〿぀-ヿ㐀-䶿一-鿿가-힯＀-￯Ѐ-ӿ֐-׿؀-ۿ]")


_ALLOW = re.compile(r"english-only:\s*allow")


_SELF = Path(__file__).resolve()


def findings(path: Path) -> list[tuple[int, str, str]]:
    """Every non-Latin character in *path*, as (line number, name, line)."""
    if path.resolve() == _SELF:
        return []
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    found = []
    for number, line in enumerate(text.splitlines(), start=1):
        if _ALLOW.search(line):
            continue
        match = _NON_LATIN.search(line)
        if match is not None:
            char = match.group(0)
            name = unicodedata.name(char, f"U+{ord(char):04X}")
            found.append((number, f"{char!r} ({name})", line.strip()))
    return found


def main(argv: list[str]) -> int:
    if not argv:
        print(
            "usage: english_only_lint.py <path> ...  (the pre-commit hook passes the staged files)",
            file=sys.stderr,
        )
        return 2
    failed = False
    for name in argv:
        path = Path(name)
        for number, char, line in findings(path):
            failed = True
            print(f"{path}:{number}: non-Latin character {char}")
            print(f"    {line}")
    if failed:
        print(
            "\nWhat lands in this tree is English -- a reader handed this "
            "checkout and nothing else has to be able to read it. Discuss in any "
            "language; commit in English. A line that genuinely needs such a "
            "character can say 'english-only: allow'.",
            file=sys.stderr,
        )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
