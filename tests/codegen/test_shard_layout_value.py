"""Render a sharded operand's ``ShardLayout`` value for a sliced mesh.

See [runtime §2.3.2](docs/spec/runtime.md#232-layoutmeshcuh).
"""

from __future__ import annotations

import re

import pytest

from tilefoundry.codegen.cuda.tir.memory.tensor_view import render_shard_layout_value
from tilefoundry.codegen.cuda.tir.stmts.mesh_scope import mesh_type
from tilefoundry.ir.core.errors import VerifyError
from tilefoundry.ir.types import Layout, Mesh, ShardLayout, Split, Topology
from tilefoundry.ir.types.layout import ComposedLayout

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
    """The value spells the slice as CuTe's own composed layout."""
    line = _mesh_layout_line(_BLOCK[2:4, :])
    assert (
        "cute::make_composed_layout(cute::identity{}, cute::Int<64>{}, "
        "cute::make_layout(cute::make_shape(cute::Int<2>{}, cute::Int<32>{}), "
        "cute::make_stride(cute::Int<32>{}, cute::Int<1>{})))" in line
    )


def test_the_sliced_mesh_type_and_value_state_the_same_geometry() -> None:
    """Type and value round-trip through one geometry, not two readings of it."""
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
    """An identity ``outer`` names no sub-box, so no offset can start in one."""
    identity_box = Mesh(
        (Topology("thread", 128),),
        ComposedLayout(inner=Layout(shape=(4, 32), strides=(32, 1)), offset=0, outer=None),
        ("w", "t"),
    )
    with pytest.raises(NotImplementedError, match="states an identity box"):
        _mesh_layout_line(identity_box)
