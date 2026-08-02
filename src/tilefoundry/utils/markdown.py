"""What the headings of a markdown document are, and what GitHub calls them.

Two readers need this and need to agree. The `spec` command addresses a section
by the number in its heading; the reference lint resolves a link's `#fragment`
against the anchor GitHub derives from that same heading. Were each to scan for
headings itself, one would eventually count a `#` line inside a fenced block as
a section and the other would not, and the disagreement would show up as a lint
that passes on a document the CLI cannot open.

So the scan lives once, here. What each caller layers on top -- keys and
disambiguation for the CLI, anchor resolution for the lint -- is its own.

This module imports nothing: it is loaded both as part of the package and, by
the pre-commit hook, straight from its file under an interpreter where
`tilefoundry` is not installed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: A heading, at any level GitHub gives an anchor to.
_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*$")

#: A fence opens or closes a block in which `# example` is a comment.
_FENCE = re.compile(r"^\s*(```|~~~)")

#: A leading `1.` / `2.10.1` in a heading is that section's own number.
_NUMBER = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+(.*)$")

#: Inline HTML, which GitHub strips before slugging.
_INLINE_HTML = re.compile(r"<[^>]*>")

#: What GitHub discards from a heading. `\w` keeps `_`, which GitHub keeps too.
_NOT_IN_ANCHOR = re.compile(r"[^\w\- ]")


@dataclass(frozen=True)
class Heading:
    """One heading line, and the three things a reader asks of it."""

    #: How many `#` it carries.
    level: int
    #: Its text with the leading number removed, or the whole text if unnumbered.
    title: str
    #: The leading `2.10.1`, or None when the heading carries no number.
    number: str | None
    #: The fragment GitHub links it by, made unique the way GitHub does.
    anchor: str
    #: Index of the heading's own line, counting from zero.
    line: int


def anchor(text: str) -> str:
    """The fragment GitHub derives from one heading's raw text.

    Not the same as a readable slug: GitHub keeps `_` and folds each remaining
    run of punctuation away rather than to a separator, so `Instance 2 - verify`
    becomes `instance-2---verify` and `register_alias` keeps its underscore.
    """
    stripped = _INLINE_HTML.sub("", text).lower()
    return _NOT_IN_ANCHOR.sub("", stripped).replace(" ", "-")


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
        if _FENCE.match(raw):
            fenced = not fenced
            continue
        if fenced:
            continue
        matched = _HEADING.match(raw)
        if matched is None:
            continue
        body = matched.group(2)
        numbered = _NUMBER.match(body)
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


__all__ = ["Heading", "anchor", "headings"]
