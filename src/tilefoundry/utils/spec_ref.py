"""Render a spec reference short, for the one place length is a cost.

A reference is written in full —
`[runtime §1.1.2](docs/spec/runtime.md#112-weight-converter-and-prepare--forward)` —
because the target half is what breaks when a heading is renamed. In a comment
or a docstring that is the whole story: a reader sees the source, and the length
buys the check.

A refusal message is the exception. It is read by someone who hit an error, in a
terminal, and a markdown link there is noise around the one thing they need. So
the message holds the same checked reference every other site holds, and this
turns it into ``spec runtime \u00a71.1.2`` on the way out.

It reads the display text and nothing else — no file, no heading list — so it
can be called from anywhere, including `ir/`, which must not import `cli/`. It
refuses anything that is not the grammar rather than passing it through: a string
that reaches here unchecked would otherwise reach a user unchecked.
"""

from __future__ import annotations

import re

#: The grammar, whole. Anchored, because a reference with prose around it is a
#: caller's mistake and not something to quietly extract from.
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
