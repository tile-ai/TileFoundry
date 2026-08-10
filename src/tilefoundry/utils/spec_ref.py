"""Render a validated spec reference compactly for terminal errors.

Source keeps the complete Markdown link so renamed headings break lint. Error
messages need only ``spec <document> <section>``. Rendering reads the display
text without importing the CLI and rejects anything outside the full reference
grammar.
"""

from __future__ import annotations

import re

_REFERENCE = re.compile(
    r"\[(?P<doc>[a-z][\w-]*) §(?P<number>[0-9](?:\.[0-9]+)*)\]"
    r"\((?P<path>[^()#]*\.md)#(?P<anchor>[^()#]+)\)"
)


def spec_ref_render(reference: str) -> str:
    """``spec runtime \u00a71.1.2`` for a reference, from its display text alone.

    Raises `ValueError` if *reference* is not one whole reference in the
    grammar.
    """
    matched = _REFERENCE.fullmatch(reference.strip())
    if matched is None:
        raise ValueError(
            f"not a spec reference: {reference!r}; expected "
            "`[<doc> §<number>](<path>#<github-anchor>)`"
        )
    return f"spec {matched['doc']} §{matched['number']}"


__all__ = ["spec_ref_render"]
