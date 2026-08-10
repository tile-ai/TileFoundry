#!/usr/bin/env python3
"""Reject committed absolute paths tied to one machine.

Named home, scratch, and conda paths fail, including environment fallbacks.
Touched files come from pre-commit; findings report their line and produce a
nonzero exit status.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_MACHINE_PATHS = re.compile(
    r"""
    (?:^|(?<=[^\w./])|(?<=-)(?=/))  # not mid-token, so a URL or a longer word is
                                    # safe -- but a `-` immediately before the
                                    # slash is `${VAR:-/path}` or `--ckpt=-`-style
                                    # punctuation, not a token this path is part of
    (
        /(?:home|Users)/[\w.-]+    # a named home directory
      | /data\d*/(?:shared/)?[\w.-]+/[\w.-]+   # a site-local scratch mount
      | /(?:opt|usr/local)/(?:miniconda\d*|anaconda\d*|conda)\b
      | /[\w./-]*?/(?:miniconda\d*|anaconda\d*)/envs/[\w.-]+
    )
    """,
    re.VERBOSE,
)


_ALLOW = re.compile(r"no-machine-path:\s*allow")


_SELF = Path(__file__).resolve()


def findings(path: Path) -> list[tuple[int, str, str]]:
    """Every machine-specific path in *path*, as (line number, match, line)."""
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
        for match in _MACHINE_PATHS.finditer(line):
            found.append((number, match.group(1), line.strip()))
    return found


def main(argv: list[str]) -> int:
    if not argv:
        print(
            "usage: no_machine_paths_lint.py <path> ...  "
            "(the pre-commit hook passes the staged files)",
            file=sys.stderr,
        )
        return 2
    failed = False
    for name in argv:
        path = Path(name)
        for number, matched, line in findings(path):
            failed = True
            print(f"{path}:{number}: machine-specific path {matched!r}")
            print(f"    {line}")
    if failed:
        print(
            "\nA location outside this project MUST come from the caller -- an "
            "environment variable, an argument, or a config file -- and MUST NOT "
            "have one machine's path as its default. State it by failing without "
            "it. A line that genuinely only mentions such a shape can say "
            "'no-machine-path: allow'.",
            file=sys.stderr,
        )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
