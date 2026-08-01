#!/usr/bin/env python3
"""Reject a comment that narrates the process instead of the code.

A comment earns its place by saying something the surrounding lines cannot say
for themselves. "AC-4-1", "milestone M2", "as review asked" say nothing about
the code: they point at a document the reader does not have, which will be
merged and deleted, after which the comment is a dangling reference that still
looks authoritative. The reason a line exists belongs in the line, the commit
message, or the spec -- three places that outlive the plan that prompted it.

The patterns here were calibrated against the tree rather than guessed. Two
tempting rules are deliberately absent:

* A bare `plan` is a domain term in this project -- a schedule plan -- and every
  one of the fourteen comments mentioning it describes the code. Only a plan
  *reference* (`plan 18`, `docs/plans/...`) is matched.
* History words (`no longer`, `previously`, `used to`) read as narration but in
  this tree describe current logic: "a function that no longer exists" is about
  a stale cache entry, not about an edit.

Run over the files a commit touches (the pre-commit hook passes them); the exit
status is non-zero if any was found.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

#: Process references that cannot mean anything to a reader of the code alone.
_NARRATION: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bAC-\d+-\d+\b"), "an acceptance-criterion id"),
    (re.compile(r"\bmilestones?\s+M\d+\b", re.IGNORECASE), "a milestone reference"),
    (re.compile(r"\bplans?\s+`?\d+`?\b", re.IGNORECASE), "a plan reference"),
    (re.compile(r"docs/plans/"), "a plan path"),
    (re.compile(r"\b(?:PR|pull request|issue)\s*#?\d+\b", re.IGNORECASE), "a PR/issue reference"),
    (re.compile(r"\bcommit\s+[0-9a-f]{7,40}\b", re.IGNORECASE), "a commit hash"),
    (
        re.compile(
            r"\b(?:as discussed|per review|review(?:er)? (?:said|asked|wants|requested)"
            r"|address(?:ed|ing) (?:the )?(?:review|feedback)|as agreed)\b",
            re.IGNORECASE,
        ),
        "review narration",
    ),
]

#: A `#` inside a string literal is not a comment. This is deliberately crude --
#: it takes the first `#` that has no quote before it on the line, which is wrong
#: only for a line whose comment follows a quote-containing expression. Such a
#: line is skipped rather than mis-parsed: a linter that invents violations is
#: worse than one that misses a few.
_COMMENT = re.compile(r"^(?P<code>[^'\"#]*)#(?P<body>.*)$")

#: A line that legitimately names one of these shapes says so.
_ALLOW = re.compile(r"comment-hygiene:\s*allow")

#: This file states the patterns, so its own source is not scanned for them.
_SELF = Path(__file__).resolve()


def findings(path: Path) -> list[tuple[int, str, str]]:
    """Every narrating comment in *path*, as (line number, what, line)."""
    if path.resolve() == _SELF or path.suffix != ".py":
        return []
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []  # unreadable or binary: nothing to claim about it
    found = []
    for number, line in enumerate(text.splitlines(), start=1):
        if _ALLOW.search(line):
            continue
        match = _COMMENT.match(line)
        if match is None:
            continue
        body = match.group("body")
        for pattern, what in _NARRATION:
            if pattern.search(body):
                found.append((number, what, line.strip()))
                break
    return found


def main(argv: list[str]) -> int:
    if not argv:
        print(
            "usage: comment_hygiene_lint.py <path> ...  "
            "(the pre-commit hook passes the staged files)",
            file=sys.stderr,
        )
        return 2
    failed = False
    for name in argv:
        path = Path(name)
        for number, what, line in findings(path):
            failed = True
            print(f"{path}:{number}: comment carries {what}")
            print(f"    {line}")
    if failed:
        print(
            "\nA comment MUST describe the code around it. Why a line exists "
            "belongs in the line, the commit message, or the spec -- the plan it "
            "came from will be merged and deleted. A line that genuinely names "
            "one of these shapes can say 'comment-hygiene: allow'.",
            file=sys.stderr,
        )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
