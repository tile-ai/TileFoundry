"""Enumerate tables and coordinates from [shard §6.1](docs/spec/shard.md#61-placement-semantics)."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from math import prod
from multiprocessing import get_context

from tilefoundry.ir import expr, types, utils

LEVELS = ("gpu", "cta", "thread")
STORAGES = (expr.StorageKind.GMEM, expr.StorageKind.SMEM, expr.StorageKind.RMEM)
OWNERS = expr.StorageOwners(LEVELS, dict(zip(STORAGES, LEVELS, strict=True)))
ATTRS = (types.S(0), types.S(1), types.B(), types.P("sum"), types.V())
SHAPE = (4, 4)


def reference(extents, attrs, selected):
    """Enumerate each unit's coordinates and value states without layout helpers."""
    attrs = attrs or (types.V(),) * len(extents)
    if any(
        SHAPE[axis] % prod(n for n, a in zip(extents, attrs, strict=True) if a == types.S(axis))
        for axis in range(2)
    ):
        raise ValueError("split extents do not divide the logical shape")
    divisors = tuple(
        prod(
            n
            for i, (n, a) in enumerate(zip(extents, attrs, strict=True))
            if a == types.S(axis) and i <= selected
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
                    if attr == types.S(axis) and i <= selected:
                        index = index * extents[i] + unit[i]
                width = SHAPE[axis] // divisors[axis]
                accepted &= coord[axis] // width == index
            if accepted:
                coordinates.add(coord)
        held[unit] = (
            coordinates,
            tuple(
                ("partial", a.reduction)
                if isinstance(a, type(types.P()))
                else "same"
                if a == types.B() or (a == types.V() and i > selected)
                else "varying"
                if a == types.V()
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
        topologies = tuple(
            types.Topology(level, n) for level, n in zip(LEVELS, extents, strict=True)
        )
        mesh = types.Mesh(topologies, types.Layout(tuple((n,) for n in extents)))
        selected = LEVELS.index(OWNERS.owner[storage])
        expected = _outcome(reference, extents, attrs, selected)
        try:
            layout = (
                types.ShardLayout.from_logical(SHAPE, mesh, attrs, storage=storage, owners=OWNERS)
                if attrs
                else types.Layout(SHAPE)
            )
            tensor = types.TensorType(
                shape=SHAPE, dtype=expr.DType.f32, layout=layout, storage=storage
            )
            actual = utils.local_type_of(
                tensor, topology_level=LEVELS[selected], topologies=topologies
            )
            if attrs:
                logical_axes = types.int_tuple.mode_axes(
                    types.int_tuple.compatible_profile(layout.layout.shape, SHAPE)
                )
                owner_shape = tuple(
                    prod(n for n, a in zip(actual.shape, logical_axes, strict=True) if a == axis)
                    for axis in range(2)
                )
            else:
                owner_shape = actual.shape
            work = utils.local_type_of(tensor)
        except ValueError:
            if expected != "error":
                failures.append(f"{extents, storage, attrs}: implementation rejected a valid case")
            continue
        if expected == "error":
            failures.append(f"{extents, storage, attrs}: implementation accepted an invalid case")
            continue
        value_kinds = {
            types.B(): "same",
            types.V(): "varying",
            types.S(0): "split",
            types.S(1): "split",
            types.P("sum"): ("partial", "sum"),
        }
        actual_states = tuple(
            "split" if isinstance(a, types.Split) else value_kinds[a]
            for a in utils.attrs_by_axis(tensor, mesh, OWNERS).values()
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
    """Literal table indices are types.B, types.V, types.S, types.P.

    Coordinate cases run in separate processes.
    """
    combine = (
        ("b", "v", "s", "partial"),
        ("v", "v", "s", "error"),
        ("s", "s", "s", "error"),
        ("partial", "error", "error", "joint"),
    )
    states = (
        types.B(),
        types.V(),
        types.S(0),
        types.S(1),
        types.P("sum"),
        types.P("max"),
        types.P("min"),
    )
    categories = (0, 1, 2, 2, 3, 3, 3)
    for (i, a), (j, b) in product(enumerate(states), repeat=2):
        for kind in (*expr.BinaryKind, None):
            cell = combine[categories[i]][categories[j]]
            expected = {"b": types.B(), "v": types.V(), "s": a if categories[i] == 2 else b}.get(
                cell, "error"
            )
            if cell == "partial":
                partial = a if categories[i] == 3 else b
                allowed = {expr.BinaryKind.MUL: {"sum"}, expr.BinaryKind.ADD: {"max", "min"}}
                if partial.reduction in allowed.get(kind, set()):
                    expected = partial
            if cell == "joint" and kind is expr.BinaryKind.ADD and a == b == types.P("sum"):
                expected = types.P("sum")
            assert _outcome(types.combine_attrs, kind, a, b) == expected, (kind, a, b)
    for attr in (*states, None):
        for kind in (*expr.BinaryKind, None):
            assert _outcome(types.combine_attrs, kind, None, attr) == "error"
            assert _outcome(types.combine_attrs, kind, attr, None) == "error"

    cases = list(product(product((1, 2), repeat=3), STORAGES, (*product(ATTRS, repeat=3), ())))
    workers = min(8, os.cpu_count() or 1)
    chunks = [cases[i::workers] for i in range(workers)]
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
        failures = [failure for part in pool.map(_check_chunk, chunks) for failure in part]
    assert not failures, "\n".join(failures)
