"""Which part is one instance's, held against the type side's answer.

``local_layout`` narrows a real tensor and ``local_type_of`` narrows a Type,
by the same rule read from the same ``ShardLayout``. They used to be checked
against each other on every single read, which put a per-call assert on the
weight path to guard a property of two implementations rather than of any
input. It is that property, so it is checked here: over a matrix of layouts,
and over every instance of each, not once per program that happens to run.
"""

from __future__ import annotations

from collections import Counter
from itertools import product

import pytest

from tilefoundry.ir.types import (
    DType,
    Mesh,
    Topology,
    make_mesh,
    make_shard_tensor_type,
    shard_layout_of,
)
from tilefoundry.ir.types.local import local_layout, local_layout_and_offset
from tilefoundry.ir.types.shard_layout import Broadcast, Split
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


def _instances(mesh: Mesh) -> int:
    """How many programs the mesh spreads a tensor over."""
    count = 1
    for extent in mesh.positions.shape:
        count *= extent
    return count


def _elements(layout, offset: int) -> list[int]:
    """Every element of the whole tensor's engine this instance reaches."""
    return [
        offset
        + sum(index * stride for index, stride in zip(coord, layout.strides, strict=True))
        for coord in product(*(range(extent) for extent in layout.shape))
    ]


@pytest.mark.parametrize("case", sorted(_CASES), ids=lambda name: name)
def test_the_layout_and_the_type_narrow_the_same_way(case: str) -> None:
    """Every instance's layout holds the extent the Type says one shard is."""
    shape, mesh, attrs = _CASES[case]
    held = make_shard_tensor_type(shape, mesh=mesh, attrs=attrs, dtype=DType.f32)
    shard = shard_layout_of(held.layout)
    want = tuple(local_type_of(held).shape)

    for program_id in range(_instances(mesh)):
        layout = local_layout(shard, shape, (program_id,))
        assert tuple(layout.shape) == want, (
            f"{case}: instance {program_id} holds {tuple(layout.shape)}, "
            f"while one shard of {shape} is {want}"
        )


@pytest.mark.parametrize("case", sorted(_CASES), ids=lambda name: name)
def test_every_instance_together_tile_the_tensor(case: str) -> None:
    """No element is held twice and none is held by nobody.

    Counted over the elements each instance reaches rather than over its
    extents, which is the property the extents alone cannot state: two
    instances could each hold the right amount and both hold it from the same
    place, and an offset that walks off the tensor is only visible once the
    elements it lands on are named.
    """
    shape, mesh, attrs = _CASES[case]
    held = make_shard_tensor_type(shape, mesh=mesh, attrs=attrs, dtype=DType.f32)
    shard = shard_layout_of(held.layout)

    covered: Counter[int] = Counter()
    for program_id in range(_instances(mesh)):
        layout, offset = local_layout_and_offset(shard, shape, (program_id,))
        covered.update(_elements(layout, offset))

    whole = 1
    for extent in shape:
        whole *= extent
    assert set(covered) == set(range(whole)), (
        f"{case}: the instances reach {sorted(set(covered))}, not the {whole} "
        f"elements of {shape}"
    )
    times = set(covered.values())
    assert len(times) == 1, f"{case}: elements held {sorted(times)} times over"


def test_the_layout_keeps_the_whole_tensors_strides() -> None:
    """A slice is the same rows the same distance apart, begun further in."""
    mesh = make_mesh((2,), ("g",), topology=_GPU)
    held = make_shard_tensor_type((4, 4), mesh=mesh, attrs=(Split(1),), dtype=DType.f32)
    shard = shard_layout_of(held.layout)

    assert local_layout_and_offset(shard, (4, 4), (0,))[0].strides == (4, 1)
    assert [local_layout_and_offset(shard, (4, 4), (gpu,))[1] for gpu in (0, 1)] == [0, 2]


def test_the_outer_axis_steps_over_what_the_inner_one_holds() -> None:
    """32 elements over a 4x8 mesh leave one each, so instance ``i`` holds it.

    One step of the outer axis clears the eight the inner one holds, rather
    than the whole the outer axis was already narrowed out of.
    """
    mesh = make_mesh((4, 8), ("w", "t"), topology=_THREAD)
    held = make_shard_tensor_type(
        (32,), mesh=mesh, attrs=(Split(0), Split(0)), dtype=DType.f32
    )
    shard = shard_layout_of(held.layout)

    offsets = [local_layout_and_offset(shard, (32,), (pid,))[1] for pid in range(32)]
    assert offsets == list(range(32))


def test_a_level_with_no_id_is_left_whole() -> None:
    """A level the host did not place divides nothing; the device does that."""
    mesh = make_mesh((2, 4), ("g", "c"), topology=_GPU)
    held = make_shard_tensor_type(
        (8, 4), mesh=mesh, attrs=(Split(0), Split(1)), dtype=DType.f32
    )
    shard = shard_layout_of(held.layout)

    layout, offset = local_layout_and_offset(shard, (8, 4), (None,))
    assert (tuple(layout.shape), offset) == ((8, 4), 0)


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
        local_layout_and_offset(shard, (6,), (0,))
