"""Enumerate tables and coordinates from [shard §6.1](docs/spec/shard.md#61-placement-semantics)."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from math import prod
from multiprocessing import get_context

from tilefoundry.ir.core.kinds import BinaryKind
from tilefoundry.ir.types import (
    B,
    DType,
    Layout,
    Mesh,
    P,
    S,
    ShardAttr,
    ShardLayout,
    Split,
    StorageKind,
    TensorType,
    Topology,
    V,
)
from tilefoundry.ir.types.int_tuple import compatible_profile, mode_axes
from tilefoundry.ir.types.storage import StorageOwners
from tilefoundry.ir.types.utils import attrs_by_axis, local_type_of

LEVELS = ("gpu", "cta", "thread")
STORAGES = (StorageKind.GMEM, StorageKind.SMEM, StorageKind.RMEM)
OWNERS = StorageOwners(LEVELS, dict(zip(STORAGES, LEVELS, strict=True)))
ATTRS = (S(0), S(1), B(), P("sum"), V())
SHAPE = (4, 4)


def reference(extents, attrs, selected):
    """Enumerate each unit's coordinates and value states without layout helpers."""
    attrs = attrs or (V(),) * len(extents)
    if any(
        SHAPE[axis] % prod(n for n, a in zip(extents, attrs, strict=True) if a == S(axis))
        for axis in range(2)
    ):
        raise ValueError("split extents do not divide the logical shape")
    divisors = tuple(
        prod(
            n
            for i, (n, a) in enumerate(zip(extents, attrs, strict=True))
            if a == S(axis) and i <= selected
        )
        for axis in range(2)
    )
    held = {}
    for unit in product(*(range(n) for n in extents)):
        coordinates = set()
        for coord in product(*(range(n) for n in SHAPE)):
            accepted = True
            for axis in range(2):
                index = 0
                for i, attr in enumerate(attrs):
                    if attr == S(axis) and i <= selected:
                        index = index * extents[i] + unit[i]
                width = SHAPE[axis] // divisors[axis]
                accepted &= coord[axis] // width == index
            if accepted:
                coordinates.add(coord)
        held[unit] = (
            coordinates,
            tuple(
                ("partial", a.reduction)
                if isinstance(a, type(P()))
                else "same"
                if a == B() or (a == V() and i > selected)
                else "varying"
                if a == V()
                else "split"
                for i, a in enumerate(attrs)
            ),
        )
    return held


def _outcome(fn, *args):
    try:
        return fn(*args)
    except ValueError:
        return "error"


def _check_chunk(cases):
    """Compare owner coordinate unions and per-unit work with local projections."""
    failures = []
    for extents, storage, attrs in cases:
        topologies = tuple(Topology(level, n) for level, n in zip(LEVELS, extents, strict=True))
        mesh = Mesh(topologies, Layout(tuple((n,) for n in extents)))
        selected = LEVELS.index(OWNERS.owner[storage])
        expected = _outcome(reference, extents, attrs, selected)
        try:
            layout = (
                ShardLayout.default(SHAPE, mesh, attrs, storage=storage, owners=OWNERS)
                if attrs
                else Layout(SHAPE)
            )
            tensor = TensorType(shape=SHAPE, dtype=DType.f32, layout=layout, storage=storage)
            actual = local_type_of(tensor, topology_level=LEVELS[selected], topologies=topologies)
            if attrs:
                logical_axes = mode_axes(compatible_profile(layout.layout.shape, SHAPE))
                owner_shape = tuple(
                    prod(n for n, a in zip(actual.shape, logical_axes, strict=True) if a == axis)
                    for axis in range(2)
                )
            else:
                owner_shape = actual.shape
            work = local_type_of(tensor)
        except ValueError:
            if expected != "error":
                failures.append(f"{extents, storage, attrs}: implementation rejected a valid case")
            continue
        if expected == "error":
            failures.append(f"{extents, storage, attrs}: implementation accepted an invalid case")
            continue
        value_kinds = {
            B(): "same",
            V(): "varying",
            S(0): "split",
            S(1): "split",
            P("sum"): ("partial", "sum"),
        }
        actual_states = tuple(
            "split" if isinstance(a, Split) else value_kinds[a]
            for a in attrs_by_axis(tensor, mesh, OWNERS).values()
        )
        for coordinates, states in expected.values():
            if actual_states != states:
                failures.append(f"{storage, attrs}: attrs {actual_states} != {states}")
                break
            shape = tuple(len({c[axis] for c in coordinates}) for axis in range(2))
            if owner_shape != shape:
                failures.append(f"{extents, storage, attrs}: owner {owner_shape} != {shape}")
                break
        per_unit = reference(extents, attrs, 2)
        for coordinates, _states in per_unit.values():
            shape = tuple(len({c[axis] for c in coordinates}) for axis in range(2))
            if work.shape != shape:
                failures.append(f"{extents, storage, attrs}: unit {work.shape} != {shape}")
                break
    return failures


def test_placement_semantics():
    """Literal table indices are B, V, S, P; coordinate cases run in separate processes."""
    combine = (
        ("b", "v", "s", "partial"),
        ("v", "v", "s", "error"),
        ("s", "s", "s", "error"),
        ("partial", "error", "error", "joint"),
    )
    states = (B(), V(), S(0), S(1), P("sum"), P("max"), P("min"))
    categories = (0, 1, 2, 2, 3, 3, 3)
    for (i, a), (j, b) in product(enumerate(states), repeat=2):
        for kind in (*BinaryKind, None):
            cell = combine[categories[i]][categories[j]]
            expected = {"b": B(), "v": V(), "s": a if categories[i] == 2 else b}.get(cell, "error")
            if cell == "partial":
                partial = a if categories[i] == 3 else b
                allowed = {BinaryKind.MUL: {"sum"}, BinaryKind.ADD: {"max", "min"}}
                if partial.reduction in allowed.get(kind, set()):
                    expected = partial
            if cell == "joint" and kind is BinaryKind.ADD and a == b == P("sum"):
                expected = P("sum")
            assert _outcome(ShardAttr.combine, kind, a, b) == expected, (kind, a, b)
    for attr in (*states, None):
        for kind in (*BinaryKind, None):
            assert _outcome(ShardAttr.combine, kind, None, attr) == "error"
            assert _outcome(ShardAttr.combine, kind, attr, None) == "error"

    cases = list(product(product((1, 2), repeat=3), STORAGES, (*product(ATTRS, repeat=3), ())))
    workers = min(8, os.cpu_count() or 1)
    chunks = [cases[i::workers] for i in range(workers)]
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
        failures = [failure for part in pool.map(_check_chunk, chunks) for failure in part]
    assert not failures, "\n".join(failures)
