"""Scan Markdown headings using the same rules as GitHub anchors.

The spec CLI and reference lint share this dependency-free scan so fenced
hashes, section numbers, and repeated-anchor suffixes cannot diverge. The lint
can load this module directly without importing ``tilefoundry``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = ["Heading", "anchor", "headings"]

HEADING_LINE = re.compile(r"^(#{1,6})\s+(.*?)\s*$")

FENCE_LINE = re.compile(r"^\s*(```|~~~)")

NUMBERED_TITLE = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+(.*)$")

INLINE_HTML = re.compile(r"<[^>]*>")

NON_ANCHOR_CHARS = re.compile(r"[^\w\- ]")


@dataclass(frozen=True)
class Heading:
    """One heading line, and the three things a reader asks of it."""

    level: int
    title: str
    number: str | None
    anchor: str
    line: int


def anchor(text: str) -> str:
    """The fragment GitHub derives from one heading's raw text.

    Not the same as a readable slug: GitHub keeps `_` and folds each remaining
    run of punctuation away rather than to a separator, so `Instance 2 - verify`
    becomes `instance-2---verify` and `register_alias` keeps its underscore.
    """
    stripped = INLINE_HTML.sub("", text).lower()
    return NON_ANCHOR_CHARS.sub("", stripped).replace(" ", "-")


def headings(text: str) -> tuple[Heading, ...]:
    """Every heading of *text*, in document order, fences skipped.

    A repeated anchor takes GitHub's `-1`, `-2` suffix. Only unnumbered headings
    can repeat -- a number is part of the slug -- so the suffix decides nothing
    for a numbered reference; it keeps the first heading the owner of the bare
    anchor rather than the last.
    """
    found: list[Heading] = []
    seen: dict[str, int] = {}
    fenced = False
    for line, raw in enumerate(text.splitlines()):
        if FENCE_LINE.match(raw):
            fenced = not fenced
            continue
        if fenced:
            continue
        matched = HEADING_LINE.match(raw)
        if matched is None:
            continue
        body = matched.group(2)
        numbered = NUMBERED_TITLE.match(body)
        base = anchor(body)
        repeat = seen.get(base, 0)
        seen[base] = repeat + 1
        found.append(
            Heading(
                level=len(matched.group(1)),
                title=numbered.group(2) if numbered else body,
                number=numbered.group(1) if numbered else None,
                anchor=base if not repeat else f"{base}-{repeat}",
                line=line,
            )
        )
    return tuple(found)
