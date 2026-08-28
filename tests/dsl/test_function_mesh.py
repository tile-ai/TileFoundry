"""Function-decorator Mesh ownership and parser scope."""

from __future__ import annotations

import pytest

from tilefoundry import func, module
from tilefoundry.analysis.walk import collect_exprs
from tilefoundry.dsl import Mesh, Tensor, Topology, tf
from tilefoundry.ir.core import Call, ExecutionDomainMetadata, get_metadata
from tilefoundry.ir.types.shard import Layout, ShardLayout
from tilefoundry.parser import ParseError
from tilefoundry.target import CudaTarget

_OUTER = Mesh(
    (Topology("cta", 8),),
    Layout(shape=(8,), strides=(1,)),
    names=("b",),
)


@module(
    entry="root",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 8), Topology("thread", 2)),
)
class DecoratedMesh:
    @func(mesh=Mesh(("cta",), (8,), names=("b",)))
    def root(x: Tensor[(8 @ mesh.b, 8), "f32", "smem"]):  # noqa: F821
        outer = tf.add(x, x)
        with Mesh(("thread",), (2,), names=("t",)) as _inner:
            nested = tf.mul(outer, outer)
            return tf.add(nested, nested)


def test_decorator_mesh_covers_signature_body_and_nested_scope() -> None:
    root = DecoratedMesh.entry_function()
    param_layout = root.params[0].type.layout
    assert isinstance(param_layout, ShardLayout)
    assert param_layout.mesh == _OUTER

    domains = [
        get_metadata(expr, ExecutionDomainMetadata)
        for expr in collect_exprs(root.body)
        if isinstance(expr, Call)
    ]
    assert domains and all(domain is not None for domain in domains)
    assert all(domain.scopes[0] == _OUTER for domain in domains)
    assert any(
        len(domain.scopes) == 2
        and domain.scopes[1].topologies == (Topology("thread", 2),)
        for domain in domains
    )


def test_decorator_mesh_resolves_the_modules_topology_extent() -> None:
    @module(
        entry="root",
        target=CudaTarget("nvidia.h200_sxm"),
        topologies=(Topology("cta", 132),),
    )
    class DeclaredExtent:
        @func(mesh=Mesh(("cta",), (8,), names=("b",)))
        def root(x: Tensor[(8 @ mesh.b,), "f32"]):  # noqa: F821
            return tf.add(x, x)

    root = DeclaredExtent.entry_function()
    domains = [
        get_metadata(expr, ExecutionDomainMetadata)
        for expr in collect_exprs(root.body)
        if isinstance(expr, Call)
    ]
    assert domains and domains[0].scopes[0].topologies == (Topology("cta", 132),)


def test_decorator_mesh_rejects_a_topology_the_module_did_not_declare() -> None:
    with pytest.raises(ParseError, match="topology 'cta' not declared by @module"):

        @module(entry="root")
        class MissingTopology:
            @func(mesh=Mesh(("cta",), (8,), names=("b",)))
            def root(x: Tensor[(8 @ mesh.b,), "f32"]):  # noqa: F821
                return tf.add(x, x)


def test_mesh_sugar_normalizes_its_layout_before_parser_binding() -> None:
    mesh = Mesh(("cta",), (2, 4), names=("x", "y"))
    assert mesh.topologies == ("cta",)
    assert mesh.layout == Layout(shape=(2, 4), strides=(4, 1))


def test_bare_undeclared_mesh_name_is_a_parse_error() -> None:
    with pytest.raises(ParseError, match="'nope' is not an active Mesh"):

        @func(
            topologies=(Topology("cta", 8),),
            mesh=Mesh(("cta",), (8,), names=("b",)),
        )
        def refused(x: Tensor[(8 @ nope,), "f32"]):  # noqa: F821
            return tf.add(x, x)
