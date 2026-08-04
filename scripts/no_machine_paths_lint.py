#!/usr/bin/env python3
"""Reject a path that only exists on the machine it was written on.

An absolute path into somebody's home directory, a shared scratch mount, or a
named conda prefix is a fact about one checkout, not about this project. Committed,
it does three things: it tells every other reader to look somewhere that is not
there, it publishes an account name and a directory layout, and -- when it sits
under `src/` -- it ships.

A default is the usual way this arrives: reading a location from the environment
and falling back to the author's own directory looks configurable and behaves like
a hardcoded path for everyone else. So the fallback is refused too. Somewhere the
caller must supply is stated by failing without it, which is the difference between
a setting and a guess.

Run over the files a commit touches (the pre-commit hook passes them); a path is
reported with the line it is on, and the exit status is non-zero if any was found.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

#: Absolute locations that belong to one machine or one account.
#:
#: Each names a *root* whose contents are outside this project: a user's home, a
#: site-local mount, an installed environment prefix. A relative path is never
#: matched -- pointing inside the repository is what paths here are for.
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

#: Lines that legitimately mention such a shape without depending on it: this
#: checker's own patterns, and a line explicitly marked as an example.
_ALLOW = re.compile(r"no-machine-path:\s*allow")

#: This file states the patterns, so its own source is not scanned for them.
_SELF = Path(__file__).resolve()


def findings(path: Path) -> list[tuple[int, str, str]]:
    """Every machine-specific path in *path*, as (line number, match, line)."""
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
