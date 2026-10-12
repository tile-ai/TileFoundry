"""A function's source declaration and definition before linking."""

from dataclasses import dataclass


@dataclass(frozen=True)
class LinkableFunction:
    """One lowered function's pre-link source, in both of its positions.

    ``name`` is the function's IR name. ``declaration`` is what a caller has
    to see and ``definition`` is the implementation; both are written from the
    one ``CallableSignature`` the compile settled for this function, so the
    two cannot come to disagree.
    """

    name: str
    declaration: str
    definition: str


__all__ = ["LinkableFunction"]
