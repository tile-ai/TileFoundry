"""Parser contracts for explicit variadic input sequences."""

from __future__ import annotations

import pytest

from tilefoundry.dsl import *
from tilefoundry.ir.core import Call, Constant, Tuple, VerifyError
from tilefoundry.ir.hir.loop_region import LoopRegion
from tilefoundry.ir.hir.mesh_region import MeshRegion
from tilefoundry.ir.hir.nn.matmul import MatMul
from tilefoundry.ir.hir.sharding.reshard import Reshard
from tilefoundry.ir.hir.tensor.tuple_get_item import TupleGetItem
from tilefoundry.parser import ParseError
from tilefoundry_nvidia import CudaTarget

_EXTERNAL_PLACEMENT_MESH = Mesh(("cta",), (2,), names=("tile",))


def test_placement_at_and_matmul_at_coexist_in_one_function() -> None:
    """The two spellings of ``@`` are told apart by position, not by shape."""

    @module(
        entry="placed_matmul",
        target=CudaTarget("nvidia.h200_sxm"),
        topologies=(Topology("cta", 2),),
    )
    class PlacedMatMul:
        @func
        def placed_matmul(x: Tensor[(2, 4), "bf16"], b: Tensor[(4, 3), "bf16"]):
            with Mesh(("cta",), (2,), ("tile",)) as mesh:
                x_local = tf.reshard(x, (2 @ mesh.tile, 4), "gmem")
                b_local = tf.reshard(b, (4, 3), "gmem")
                return x_local @ b_local

    scope = PlacedMatMul.entry_function().body
    assert isinstance(scope, MeshRegion)
    body = scope.body
    assert isinstance(body, Call)
    assert isinstance(body.target, MatMul)
    assert any(isinstance(argument.target, Reshard) for argument in body.args)


def test_value_less_mesh_region_exposes_multiple_values_through_a_tuple() -> None:
    """A value-less with exports only its live values through one region result."""

    @module(
        entry="multi_escape",
        target=CudaTarget("nvidia.h200_sxm"),
        topologies=(Topology("cta", 2),),
    )
    class MultiEscape:
        @func
        def multi_escape(x: Tensor[(2,), "f32"]):
            with Mesh(("cta",), layout=(2,), names=("tile",)) as _mesh:
                first = tf.relu(x)
                second = tf.relu(first)
            return first, second

    body = MultiEscape.entry_function().body
    assert isinstance(body, Tuple)
    first, second = body.elements
    assert isinstance(first, Call) and isinstance(second, Call)
    assert isinstance(first.target, TupleGetItem)
    assert isinstance(second.target, TupleGetItem)
    scope = first.args[0]
    assert isinstance(scope, MeshRegion)
    assert isinstance(scope.body, Tuple)
    assert len(scope.body.elements) == 2
    assert isinstance(first.args[1], Constant) and first.args[1].value == 0
    assert isinstance(second.args[1], Constant) and second.args[1].value == 1


def test_valueful_mesh_region_also_wraps_escaping_bindings() -> None:
    """A with result and an escaping assignment keep both region edges."""

    @module(
        entry="valueful_escape",
        target=CudaTarget("nvidia.h200_sxm"),
        topologies=(Topology("cta", 2),),
    )
    class ValuefulEscape:
        @func
        def valueful_escape(x: Tensor[(2,), "f32"]):
            with Mesh(("cta",), layout=(2,), names=("tile",)) as _mesh:
                value = tf.relu(x)
                for _index in tf.tile(2, 1):  # noqa: F821
                    value = tf.relu(value)
            return value

    scoped = ValuefulEscape.entry_function().body
    assert isinstance(scoped, MeshRegion)
    assert isinstance(scoped.body, LoopRegion)
    assert isinstance(scoped.body.body, Call)


def test_mesh_binding_does_not_escape_its_with_scope() -> None:
    """A mesh alias is removed with its lexical frame after the with body."""
    with pytest.raises(ParseError, match="'mesh' is not a lexical Mesh binding"):

        @module(
            entry="escaped_mesh",
            target=CudaTarget("nvidia.h200_sxm"),
            topologies=(Topology("cta", 2),),
        )
        class EscapedMesh:
            @func
            def escaped_mesh(x: Tensor[(2,), "f32"]):
                with Mesh(("cta",), layout=(2,), names=("tile",)) as mesh:
                    value = tf.relu(x)
                return tf.reshard(value, (2 @ mesh.tile,), "gmem")


def test_a_boundary_rejects_a_different_split_axis() -> None:
    with pytest.raises(VerifyError, match="type mismatch.*param 'x'"):

        @module(
            entry="root",
            target=CudaTarget("nvidia.h200_sxm"),
            topologies=(Topology("cta", 8),),
        )
        class AxisMismatch:
            @func(mesh=Mesh(("cta",), (8,), names=("b",)))
            def stage(x: Tensor[(8, 8 @ mesh.b), "f32", "smem"]):  # noqa: F821
                return tf.add(x, x)

            @func(mesh=Mesh(("cta",), (8,), names=("b",)))
            def root(x: Tensor[(8, 8), "f32"]):
                return stage(  # noqa: F821
                    tf.reshard(x, (8 @ mesh.b, 8), "smem")  # noqa: F821
                )


def test_a_boundary_rejects_a_different_storage() -> None:
    with pytest.raises(VerifyError, match="type mismatch.*param 'x'"):

        @module(
            entry="root",
            target=CudaTarget("nvidia.h200_sxm"),
            topologies=(Topology("cta", 8),),
        )
        class StorageMismatch:
            @func(mesh=Mesh(("cta",), (8,), names=("b",)))
            def stage(x: Tensor[(8 @ mesh.b, 8), "f32", "rmem"]):  # noqa: F821
                return tf.add(x, x)

            @func(mesh=Mesh(("cta",), (8,), names=("b",)))
            def root(x: Tensor[(8, 8), "f32"]):
                return stage(  # noqa: F821
                    tf.reshard(x, (8 @ mesh.b, 8), "smem")  # noqa: F821
                )


def test_a_layout_undefined_boundary_still_constrains_storage() -> None:
    with pytest.raises(VerifyError, match="type mismatch.*param 'x'"):

        @module(
            entry="root",
            target=CudaTarget("nvidia.h200_sxm"),
            topologies=(Topology("cta", 8),),
        )
        class LayoutUndefinedStorageMismatch:
            @func(mesh=Mesh(("cta",), (8,), names=("b",)))
            def stage(x: Tensor[(8, 8), "bf16", None, "smem"]):
                return tf.add(x, x)

            @func(mesh=Mesh(("cta",), (8,), names=("b",)))
            def root(x: Tensor[(8, 8), "f32"]):
                actual = tf.reshard(tf.cast(x, dtype="bf16"), (8, 8), "rmem")
                return stage(actual)  # noqa: F821


def test_a_boundary_rejects_a_bare_undeclared_mesh_name() -> None:
    with pytest.raises(ParseError, match="'nope' is not a lexical Mesh binding"):

        @module(
            entry="root",
            target=CudaTarget("nvidia.h200_sxm"),
            topologies=(Topology("cta", 8),),
        )
        class UndeclaredMesh:
            @func(mesh=Mesh(("cta",), (8,), names=("b",)))
            def root(x: Tensor[(8 @ nope, 8), "f32"]):  # noqa: F821
                return tf.add(x, x)


def test_placement_rejects_an_external_mesh_axis_binding() -> None:
    with pytest.raises(
        ParseError, match="'_EXTERNAL_PLACEMENT_MESH' is not a lexical Mesh binding"
    ):

        @module(
            entry="run",
            target=CudaTarget("nvidia.h200_sxm"),
            topologies=(Topology("cta", 2),),
        )
        class ExternalMeshPlacement:
            @func
            def run(x: Tensor[(2 @ _EXTERNAL_PLACEMENT_MESH.tile,), "f32"]):  # noqa: F821
                return x
