#!/usr/bin/env python3
"""Reject non-Latin script in anything this repository commits.

`docs/SPEC-RULES.md` states it for spec text; the reason is not specific to
specs. Everything committed here is read by contributors and by agents that were
handed this checkout and nothing else: an identifier, a comment, a docstring, or
a document in a script the reader cannot type is a dead end for them. Discussion
happens in whatever language suits the people having it; what lands in the tree
is English.

Detected by script, not by language: CJK, Hangul, Cyrillic, Arabic and Hebrew
ranges. This catches the case that actually occurs -- notes written in the
author's own language -- without trying to judge English prose. Latin-1 accented
letters are left alone, so a name like `Müller` or a word like `naïve` passes.
Greek is left alone too, and deliberately: it is mathematical notation here
(`Σ`, `Π` in the layout algebra and in `spec runtime`), read the same way by
everyone.

Run over the files a commit touches (the pre-commit hook passes them); the exit
status is non-zero if any was found.
"""

from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path

#: Scripts a reader of this repository is not assumed to have.
_NON_LATIN = re.compile(
    "["
    "　-〿"  # CJK punctuation
    "぀-ヿ"  # Hiragana, Katakana
    "㐀-䶿"  # CJK extension A
    "一-鿿"  # CJK unified ideographs
    "가-힯"  # Hangul syllables
    "＀-￯"  # fullwidth and halfwidth forms
    "Ѐ-ӿ"  # Cyrillic
    "֐-׿"  # Hebrew
    "؀-ۿ"  # Arabic
    "]"
)

#: A line that legitimately carries such a character -- a test of this very
#: behaviour, an encoding fixture -- says so.
_ALLOW = re.compile(r"english-only:\s*allow")

#: This file states the ranges, so its own source is not scanned.
_SELF = Path(__file__).resolve()


def findings(path: Path) -> list[tuple[int, str, str]]:
    """Every non-Latin character in *path*, as (line number, name, line)."""
    if path.resolve() == _SELF:
        return []
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []  # unreadable or binary: nothing to claim about it
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
            "usage: english_only_lint.py <path> ...  "
            "(the pre-commit hook passes the staged files)",
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
