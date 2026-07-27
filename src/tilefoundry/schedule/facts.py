"""What the schedule layer asks a target for.

These aggregates are this layer's own vocabulary. The atom catalogue and the
store a tile lives in are target-specific, but *asking* for them must not be: a
scheduling stage names the facts it needs and the target package registers the
conversion that supplies them, so no stage calls into a target through a service
object it has to know the shape of.

The store a tile occupies belongs to the level being scheduled rather than to the
device -- an AMX tile at the ``core`` level lives in L1d, a CUDA one at the
``cta`` level in shared memory -- which is why the stage is the query.
"""

from __future__ import annotations

from dataclasses import dataclass

from tilefoundry.ir.core import Call
from tilefoundry.ir.types import DType


@dataclass(frozen=True)
class AtomFact:
    """One atom's facts, for later CP-SAT atom selection.

    ``shape``/``dtype`` mirror the atom's own MNK shape and
    ``(a, b, c)`` dtypes for a quick look without unpacking ``atom``.
    ``duration`` is a nominal roofline estimate in ns -- a placeholder to
    rank against until a measured number backfills it;
    ``compute_duration`` is its compute-side half alone, for a consumer
    that models the surrounding traffic itself and would otherwise charge
    memory twice. ``storage`` is per-thread fragment occupancy in bytes;
    ``resource`` is the required thread-scope footprint, e.g.
    ``{"lane": 32}`` for one warp. ``is_async`` marks an asynchronous
    instruction. ``atom`` is the target's own realized descriptor, carried
    through so a later fill/codegen stage need not re-resolve it from
    ``shape``/``dtype``.

    The descriptor stays opaque on purpose, so a target package can enumerate
    its own catalogue without this type knowing that catalogue's classes.
    """

    shape: tuple[int, int, int]
    dtype: tuple[DType, DType, DType]
    duration: float
    compute_duration: float
    storage: dict[str, int]
    resource: dict[str, int]
    is_async: bool
    atom: object


@dataclass(frozen=True)
class TileStoreFacts:
    """The store a tile of one scheduled level occupies.

    Projected once per solve, because the capacity is a property of the level
    and of the hardware -- not of any one operation being scheduled.
    """

    stage: str
    tile_capacity_bytes: int


@dataclass(frozen=True)
class AtomCandidateQuery:
    """Which operation's catalogue is being asked for, at which level."""

    stage: str
    op: Call


@dataclass(frozen=True)
class AtomCandidateFacts:
    """The atoms one target admits for one operation, in catalogue order.

    Order is part of the answer: it is the order the target enumerated, and a
    consumer that ranks candidates must see the same sequence every run.
    """

    candidates: tuple[AtomFact, ...]


__all__ = [
    "AtomCandidateFacts",
    "AtomCandidateQuery",
    "AtomFact",
    "TileStoreFacts",
]
