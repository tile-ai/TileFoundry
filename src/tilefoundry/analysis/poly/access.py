"""Polyhedral access and time queries."""

from __future__ import annotations

import math
from dataclasses import dataclass

import isl

from .errors import ExtractError
from .extract import (
    TileGraph,
    _as_map,
    _buffers_by_statement,
    _only_out_dim,
    _static_extent,
    _travels_with,
)


@dataclass(frozen=True)
class AxisExtent:
    """Describe one buffer dimension within a statement's complete access.

    ``extent`` measures reached elements rather than deriving a tile size.
    ``axes`` identifies time dimensions that reach them and carries no size;
    when empty, every iteration touches the dimension in full.
    """

    axes: tuple[int, ...]
    extent: int


@dataclass(frozen=True)
class AccessFootprint:
    """One access sized per buffer dimension, so the element count is the product over ``dims``.

    One (statement, buffer) access sized per buffer dimension, so the
    element count is the product over ``dims``.

    The count is the bounding box of the access's range, which is exact for a
    box-shaped access and an upper bound for one that leaves holes in it (a
    diagonal ``b[t0 + t1]`` reaches a band inside its own box).
    """

    statement: str
    buffer: str
    is_read: bool
    dims: tuple[AxisExtent, ...]
    elem_bytes: int


def time_extents(tg: TileGraph, time_map: "isl.union_map") -> tuple[int, ...]:
    """Per-dimension extent of ``time_map``'s range over ``tg.domain``.

    Raises unless every dimension starts at 0, since a tile index counts
    from the origin.
    """
    sets: list["isl.set"] = []
    time_map.intersect_domain(tg.domain).range().foreach_set(sets.append)
    if len(sets) != 1:
        raise ExtractError(
            f"time_extents: expected one time space, got {len(sets)} -- "
            "every statement must share the band's own range space"
        )
    box = sets[0]
    extents = []
    for i in range(box.dim(isl.dim_type.SET)):
        lo, hi = _static_extent(box, i, "time_extents")
        if lo != 0:
            raise ExtractError(
                f"time_extents: time dimension {i} starts at {lo}, not 0 -- "
                "tile counting assumes an origin-based extent"
            )
        extents.append(hi + 1)
    return tuple(extents)


def statement_time_dims(tg: TileGraph, time_map: "isl.union_map") -> dict[str, tuple[int, ...]]:
    """Per statement, one entry per time dimension.

    Per statement, one entry per time dimension: the statement's own
    domain dimension that dimension travels with, or ``-1`` when it is
    constant there (``RN[d0] -> [d0, 63, 127]`` gives ``(0, -1, -1)``).
    Raises on a skewed time dimension, which no per-axis tile size can
    describe.
    """
    maps: list["isl.map"] = []
    time_map.foreach_map(maps.append)
    out: dict[str, tuple[int, ...]] = {}
    for m in maps:
        name = m.get_tuple_name(isl.dim_type.IN)
        row = []
        for pos in range(m.dim(isl.dim_type.OUT)):
            involved = _travels_with(m, pos)
            if len(involved) > 1:
                raise ExtractError(
                    f"statement_time_dims: time dimension {pos} of statement "
                    f"{name!r} mixes domain dimensions {involved} ({m}) -- "
                    "a skewed band has no per-axis tile size"
                )
            row.append(involved[0] if involved else -1)
        out[name] = tuple(row)
    return out


def carried_distances(
    tg: TileGraph, time_map: "isl.union_map", n_dims: int
) -> dict[str, tuple[int, ...]]:
    """Per buffer, the largest dependence distance isl reports along each time dimension.

    Per buffer, the largest dependence distance isl reports along each
    time dimension. A flow dependence ``a -> b`` is attributed to every
    buffer ``a`` writes and ``b`` reads, which for a RAW must-dependence is
    exactly the memory it travels through.
    """
    written = _buffers_by_statement(tg.writes)
    read = _buffers_by_statement(tg.reads)
    names = {buf for bufs in (*written.values(), *read.values()) for buf in bufs}
    distances: dict[str, list[int]] = {buf: [0] * n_dims for buf in names}
    deps: list["isl.map"] = []
    tg.deps.foreach_map(deps.append)
    for dep in deps:
        carriers = written.get(dep.get_tuple_name(isl.dim_type.IN), set()) & read.get(
            dep.get_tuple_name(isl.dim_type.OUT), set()
        )
        if not carriers:
            continue
        pieces: list["isl.set"] = []
        dep.apply_domain(time_map).apply_range(time_map).deltas().foreach_set(pieces.append)
        for piece in pieces:
            for i in range(n_dims):
                lo, hi = _static_extent(piece, i, "carried_distances")
                reach = max(abs(lo), abs(hi))
                for buf in carriers:
                    distances[buf][i] = max(distances[buf][i], reach)
    return {buf: tuple(dims) for buf, dims in distances.items()}


def access_footprints(tg: TileGraph, time_map: "isl.union_map") -> tuple[AccessFootprint, ...]:
    """Access footprints.

    Every read and write of ``tg``, expressed against ``time_map``'s
    range so a tile size per time dimension sizes it (see
    :class:`AccessFootprint`).
    """
    out: list[AccessFootprint] = []
    for um, is_read in ((tg.reads, True), (tg.writes, False)):
        maps: list["isl.map"] = []
        um.foreach_map(maps.append)
        for m in maps:
            stmt = m.get_tuple_name(isl.dim_type.IN)
            buf = m.get_tuple_name(isl.dim_type.OUT)
            dtype = tg.buffer_dtypes.get(buf)
            if dtype is None:
                raise ExtractError(
                    f"access_footprints: buffer {buf!r} has no recorded dtype "
                    "-- extract must resolve every accessed buffer's element type"
                )
            timed = _as_map(m.apply_domain(time_map))
            dims = []
            for pos in range(timed.dim(isl.dim_type.OUT)):
                lo, hi = _static_extent(
                    _only_out_dim(timed, pos).range(), 0, f"access_footprints[{buf}]"
                )
                dims.append(
                    AxisExtent(axes=_travels_with(timed, pos), extent=hi - lo + 1)
                )
            out.append(
                AccessFootprint(
                    statement=stmt, buffer=buf, is_read=is_read, dims=tuple(dims),
                    elem_bytes=math.ceil(dtype.bit_width / 8),
                )
            )
    return tuple(out)

__all__ = [
    "AccessFootprint",
    "AxisExtent",
    "access_footprints",
    "carried_distances",
    "statement_time_dims",
    "time_extents",
]
