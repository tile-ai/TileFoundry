"""What the schedule layer asks a target about one instruction.

An atom is target-specific, but *asking* for one must not be: an algorithm names
the facts it needs and the target package registers the conversion that supplies
them, so nothing reaches into a target through an object whose shape it has to
know. Each algorithm family owns the rest of its own facts vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass

from tilefoundry.ir.types import DType


@dataclass(frozen=True)
class AtomFact:
    """One atom's facts, for later CP-SAT atom selection.

    Shape and dtype describe MNK and A/B/C. Durations are nominal nanoseconds;
    ``compute_duration`` excludes traffic. Storage is per-thread bytes and
    resource is thread-scope occupancy. ``atom`` remains target-owned and
    opaque so later stages can use the realized descriptor directly.
    """

    shape: tuple[int, int, int]
    dtype: tuple[DType, DType, DType]
    duration: float
    compute_duration: float
    storage: dict[str, int]
    resource: dict[str, int]
    is_async: bool
    atom: object


__all__ = ["AtomFact"]
