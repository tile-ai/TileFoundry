"""Value types a record states and Python has no shape for.

A field that a plain ``int``, ``str``, or mapping already reads correctly does
not belong here -- a name that renders exactly like ``int`` says nothing extra.
These two earn their place: a bare pair cannot say which half is the whole
machine's, and Python has no half-open interval parameterized by a trip index.

They carry values only. What they look like is decided where the rest of the
rendering is, in inspection.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TotalAndPerUnit[V]:
    """One quantity, stated for the whole machine and for one unit of it."""

    total: V
    per_unit: V


@dataclass(frozen=True)
class TripInterval:
    """A half-open interval, parameterized by the trip index of its loop.

    A single trip occurs once, so ``stride`` says nothing about it and the trip
    index does not appear. Repeated trips state the first interval and the
    stride every later one is offset by.
    """

    start: int
    end: int
    stride: int = 0
    trips: int = 1


__all__ = ["TotalAndPerUnit", "TripInterval"]
