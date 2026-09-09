"""Render a sharded operand's ``ShardLayout`` value for a sliced mesh.

A slice lives in the IR as ``ComposedLayout(inner, offset, outer)``, and the
emitted C++ has to carry both halves: the sub-box the mesh covers and where in
the launch level it starts. The type renderer already did; these pin the value
renderer to the same answer, since the runtime reads the offset back off the
mesh to turn a program id into a mesh coordinate.

See [runtime §2.3](docs/spec/runtime.md#23-tilefoundrymesh).
"""

from __future__ import annotations

import re

import pytest

from tilefoundry.codegen.cuda.tir.memory.tensor_view import render_shard_layout_value
from tilefoundry.codegen.cuda.tir.stmts.mesh_scope import mesh_type
from tilefoundry.ir.core.errors import VerifyError
from tilefoundry.ir.types.shard import Layout, Mesh, ShardLayout, Split, Topology
from tilefoundry.ir.types.shard.layout import ComposedLayout

_BLOCK = Mesh(
    (Topology("thread", 128),), Layout(shape=(4, 32), strides=(32, 1)), ("w", "t")
)


def _shard_layout(mesh: Mesh) -> ShardLayout:
    """A (4, 32) operand split both ways over *mesh*."""
    return ShardLayout(
        layout=Layout(shape=(4, 32), strides=(32, 1)),
        attrs=(Split(0), Split(1)),
        mesh=mesh,
    )


def _mesh_layout_line(mesh: Mesh) -> str:
    """The one preamble line that builds the mesh layout value."""
    preamble, _ = render_shard_layout_value("v", _shard_layout(mesh))
    (line,) = [entry for entry in preamble if entry.startswith("auto v__mesh_layout")]
    return line


def test_a_sliced_mesh_value_carries_its_offset_and_sub_box() -> None:
    """The value spells the slice as CuTe's own composed layout.

    Threads 64..127 are two of the block's four warps, so the mesh the value
    builds states the sub-box ``(2, 32)`` at the parent's strides, wrapped in
    the offset 64 that says which two. Emitting the bare positions would leave
    every instance reading the box its neighbour owns.
    """
    line = _mesh_layout_line(_BLOCK[2:4, :])
    assert (
        "cute::make_composed_layout(cute::identity{}, cute::Int<64>{}, "
        "cute::make_layout(cute::make_shape(cute::Int<2>{}, cute::Int<32>{}), "
        "cute::make_stride(cute::Int<32>{}, cute::Int<1>{})))" in line
    )


def test_the_sliced_mesh_type_and_value_state_the_same_geometry() -> None:
    """Type and value round-trip through one geometry, not two readings of it.

    ``make_shard_tensor`` takes the value and the runtime reads the type, so a
    disagreement between them is a silently wrong shard origin rather than a
    compile error. The integers are the whole geometry: offset, extents,
    strides.
    """
    sliced = _BLOCK[2:4, :]
    numbers = re.compile(r"cute::Int<(-?\d+)>")
    assert numbers.findall(_mesh_layout_line(sliced)) == ["64", "2", "32", "32", "1"]
    mesh_only = mesh_type(sliced)
    assert numbers.findall(mesh_only) == ["64", "2", "32", "32", "1"]


def test_an_unsliced_mesh_value_stays_a_plain_layout() -> None:
    """A mesh that is the whole level starts at zero, and says nothing more."""
    line = _mesh_layout_line(_BLOCK)
    assert "make_composed_layout" not in line
    assert (
        "cute::make_layout(cute::make_shape(cute::Int<4>{}, cute::Int<32>{}), "
        "cute::make_stride(cute::Int<32>{}, cute::Int<1>{}))" in line
    )


def test_a_non_contiguous_mesh_slice_is_refused() -> None:
    """A slice that is not one run of instances has no single offset to emit."""
    with pytest.raises(VerifyError, match="contiguous thread interval"):
        _mesh_layout_line(_BLOCK[:, 1:])


def test_an_identity_participating_box_is_refused() -> None:
    """An identity ``outer`` names no sub-box, so no offset can start in one.

    ``ComposedLayout.outer`` is what a mesh slice puts its participating box
    in; ``None`` there means the extents come from ``inner`` instead, a
    component a mesh has no other use for. Refusing beats emitting the inner
    component's box as if the slice had selected it.
    """
    identity_box = Mesh(
        (Topology("thread", 128),),
        ComposedLayout(inner=Layout(shape=(4, 32), strides=(32, 1)), offset=0, outer=None),
        ("w", "t"),
    )
    with pytest.raises(NotImplementedError, match="states an identity box"):
        _mesh_layout_line(identity_box)
