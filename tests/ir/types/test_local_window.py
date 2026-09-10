"""Which part is one instance's, held against the type side's answer.

``local_window`` narrows a real tensor and ``local_type_of`` narrows a Type,
by the same rule read from the same ``ShardLayout``. They used to be checked
against each other on every single read, which put a per-call assert on the
weight path to guard a property of two implementations rather than of any
input. It is that property, so it is checked here: over a matrix of layouts,
and over every instance of each, not once per program that happens to run.
"""

from __future__ import annotations

import pytest

from tilefoundry.ir.types import DType, make_shard_tensor_type
from tilefoundry.ir.types.shard import Mesh, Topology, make_mesh, shard_layout_of
from tilefoundry.ir.types.shard.local import local_window
from tilefoundry.ir.types.shard.shard_layout import Broadcast, Split
from tilefoundry.ir.types.utils import local_type_of

_GPU, _THREAD = Topology("gpu", 2), Topology("thread", 32)


def _mesh(topology: Topology, extents: tuple[int, ...], names: tuple[str, ...]) -> Mesh:
    return make_mesh(extents, names, topology=topology)


_CASES = {
    "one axis one level": ((8, 4), _mesh(_GPU, (2,), ("g",)), (Split(0),)),
    "the other axis": ((8, 4), _mesh(_GPU, (2,), ("g",)), (Split(1),)),
    "broadcast keeps it whole": ((8, 4), _mesh(_GPU, (2,), ("g",)), (Broadcast(),)),
    "two axes of one level over one tensor axis": (
        (32,),
        _mesh(_THREAD, (4, 8), ("w", "t")),
        (Split(0), Split(0)),
    ),
    "two axes of one level over two tensor axes": (
        (8, 16),
        _mesh(_THREAD, (4, 8), ("w", "t")),
        (Split(0), Split(1)),
    ),
    "one axis split, one broadcast": (
        (8, 16),
        _mesh(_THREAD, (4, 8), ("w", "t")),
        (Split(1), Broadcast()),
    ),
}


@pytest.mark.parametrize("case", sorted(_CASES), ids=lambda name: name)
def test_the_window_and_the_type_narrow_the_same_way(case: str) -> None:
    """Every instance's window is the extent the Type says one shard is."""
    shape, mesh, attrs = _CASES[case]
    held = make_shard_tensor_type(shape, mesh=mesh, attrs=attrs, dtype=DType.f32)
    shard = shard_layout_of(held.layout)
    want = tuple(local_type_of(held).shape)

    instances = 1
    for extent in mesh.layout.shape:
        instances *= extent
    for program_id in range(instances):
        window = local_window(shard, shape, (program_id,))
        assert tuple(extent for _start, extent in window) == want, (
            f"{case}: instance {program_id} holds "
            f"{tuple(extent for _s, extent in window)}, "
            f"while one shard of {shape} is {want}"
        )


@pytest.mark.parametrize("case", sorted(_CASES), ids=lambda name: name)
def test_the_windows_of_every_instance_tile_the_tensor(case: str) -> None:
    """No element is held twice and none is held by nobody.

    Which is the property the extents alone cannot state: two instances could
    each hold the right amount and both hold it from the same place.
    """
    shape, mesh, attrs = _CASES[case]
    held = make_shard_tensor_type(shape, mesh=mesh, attrs=attrs, dtype=DType.f32)
    shard = shard_layout_of(held.layout)

    instances = 1
    for extent in mesh.layout.shape:
        instances *= extent
    covered: dict[tuple[int, ...], int] = {}
    for program_id in range(instances):
        window = local_window(shard, shape, (program_id,))
        for point in _points(window):
            covered[point] = covered.get(point, 0) + 1

    whole = 1
    for extent in shape:
        whole *= extent
    assert len(covered) == whole, f"{case}: {whole - len(covered)} elements held by none"
    times = set(covered.values())
    assert len(times) == 1, f"{case}: elements held {sorted(times)} times over"


def _points(window: tuple[tuple[int, int], ...]) -> list[tuple[int, ...]]:
    """Every coordinate one window covers."""
    points: list[tuple[int, ...]] = [()]
    for start, extent in window:
        points = [
            (*point, index) for point in points for index in range(start, start + extent)
        ]
    return points


def test_a_level_with_no_id_is_left_whole() -> None:
    """A level the host did not place divides nothing; the device does that."""
    mesh = make_mesh((2, 4), ("g", "c"), topology=_GPU)
    held = make_shard_tensor_type(
        (8, 4), mesh=mesh, attrs=(Split(0), Split(1)), dtype=DType.f32
    )
    shard = shard_layout_of(held.layout)

    assert local_window(shard, (8, 4), (None,)) == ((0, 8), (0, 4))


def test_an_extent_its_mesh_axis_does_not_divide_is_refused() -> None:
    """A shard would not be one slice, so it is refused by name.

    ``canonical_shard_layout`` refuses this when the layout is built, so the
    shape asked about here is one the layout was not built for -- which is the
    only way the two can disagree, and worth naming rather than slicing.
    """
    mesh = make_mesh((4,), ("g",), topology=_GPU)
    held = make_shard_tensor_type((8,), mesh=mesh, attrs=(Split(0),), dtype=DType.f32)
    shard = shard_layout_of(held.layout)

    with pytest.raises(ValueError, match=r"extent 6, which its mesh axis of 4"):
        local_window(shard, (6,), (0,))
