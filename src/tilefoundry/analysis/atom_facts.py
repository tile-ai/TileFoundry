"""``AtomFact`` -- one candidate atom's target-relevant facts, as the
schedule layer consumes them. Target-independent on purpose: the atom
descriptor itself stays opaque (``atom``), so a target package can
enumerate its own catalogue (``target/cuda/atoms.py``) without this type
knowing that catalogue's types.
"""
from __future__ import annotations

from dataclasses import dataclass

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
