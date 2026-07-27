"""``@func(topologies=(...))`` + ``with Mesh(topology="...", ...)`` parser tests."""

from __future__ import annotations

import pytest

from tilefoundry import func
from tilefoundry.dsl import Tensor
from tilefoundry.dsl.tf import *  # noqa: F403
from tilefoundry.inspection import as_script
from tilefoundry.ir.core import Call, VerifyError
from tilefoundry.ir.hir.sharding.reshard import Reshard
from tilefoundry.ir.types.shard import (
    Layout,
    Mesh,
    S,
    ShardLayout,
    Topology,
)
from tilefoundry.parser.hir_parser import parse_script


# Canonical fixture: declares 2 topologies, uses string-name Mesh scope,
# embeds reshard ops referencing the resolved meshes (verbose form).
@func(topologies=(Topology("cta", 128), Topology("thread", 8 * 32)))
def demo_topology_canonical(
    a: Tensor[(1, 1536), "f32"],
) -> Tensor[(1, 1536), "f32"]:
    with Mesh(
        topology="cta", layout=Layout(shape=(128,), strides=(1,))
    ) as cta_mesh:
        b = reshard(  # noqa: F405
            a,
            layout=ShardLayout(
                layout=Layout(shape=(1, 1536), strides=(1536, 1)),
                attrs=(), mesh=cta_mesh,
            ),
            storage="smem",
        )
        with Mesh(
            topology="thread",
            layout=Layout(shape=(8, 32), strides=(32, 1)),
        ) as thread_mesh:
            c = reshard(  # noqa: F405
                b,
                layout=ShardLayout(
                    layout=Layout(shape=(1, 8, 192), strides=(1536, 192, 1)),
                    attrs=(S(1), S(2)), mesh=thread_mesh,
                ),
                storage="rmem",
            )
            return c
    raise RuntimeError("unreachable")


def _walk_reshard_meshes(e):
    """Yield every Reshard target mesh reachable through Call args."""
    if isinstance(e, Call):
        if isinstance(e.target, Reshard):
            yield e.target.layout.mesh
        for a in e.args:
            yield from _walk_reshard_meshes(a)


def test_topology_declarations_and_mesh_string_resolution() -> None:
    """``topologies=`` records inline declarations; string topology
    inside ``with Mesh(...)`` resolves to the declared Topology."""
    declared_topologies = demo_topology_canonical.effective_topologies()
    ir = demo_topology_canonical.entry_function()
    assert [t.name for t in declared_topologies] == ["cta", "thread"]
    assert [t.size for t in declared_topologies] == [128, 256]

    # All reshard meshes refer to declared topologies.
    declared = {t.name for t in declared_topologies}
    meshes = list(_walk_reshard_meshes(ir.body))
    assert len(meshes) >= 2
    assert all(m.topology.name in declared for m in meshes)


def test_topology_errors() -> None:
    """Duplicate topology name + unknown Mesh topology both raise."""
    with pytest.raises(VerifyError, match="duplicate topology name"):

        @func(topologies=(Topology("cta", 128), Topology("cta", 64)))
        def _dup(a: Tensor[(1, 1536), "f32"]) -> Tensor[(1, 1536), "f32"]:
            return a

    with pytest.raises(VerifyError, match="topology.*not declared"):

        @func(topologies=(Topology("cta", 128),))
        def _unk(a: Tensor[(1, 1536), "f32"]) -> Tensor[(1, 1536), "f32"]:
            with Mesh(
                topology="nonexistent",
                layout=Layout(shape=(128,), strides=(1,)),
            ) as m:  # noqa: F841
                return a


def test_topology_roundtrip_through_printer() -> None:
    """Print → parse round-trip preserves topology declarations and sizes."""
    src = as_script(demo_topology_canonical)
    assert '@module(entry="demo_topology_canonical")' in src
    assert "topologies = (" in src
    assert 'Topology("cta", 128)' in src
    reparsed = parse_script(src)
    assert reparsed.topologies == demo_topology_canonical.topologies


# Sugar form: tuple-literal mesh layout + body sugar with @-bindings.
@func(topologies=(Topology("cta", 128), Topology("thread", 8 * 32)))
def demo_topology_sugar(
    a: Tensor[(1, 1536), "f32"],
) -> Tensor[(1, 1536), "f32"]:
    with Mesh(topology="cta", layout=(128,)) as cta_mesh:  # noqa: F841
        b = reshard(a, layout=(1, 1536), storage="smem")  # noqa: F405
        with Mesh(topology="thread", layout=(8, 32)) as thread_mesh:
            c = reshard(  # noqa: F405
                b,
                layout=(1, 8 @ thread_mesh.x, 192 @ thread_mesh.y),
                storage="rmem",
            )
            return c
    raise RuntimeError("unreachable")


def test_topology_sugar_layout_parses_and_lowers_correctly() -> None:
    """Tuple-literal Mesh layout + body sugar with @-bindings parses to
    the same shape/strides as the verbose form."""
    fn = demo_topology_sugar.entry_function()
    assert fn.name == "demo_topology_sugar"
    assert len(demo_topology_sugar.effective_topologies()) == 2

    def _all_calls(e):
        out = []
        if isinstance(e, Call):
            out.append(e)
            for a in e.args:
                out.extend(_all_calls(a))
        return out

    reshards = [c for c in _all_calls(fn.body) if isinstance(c.target, Reshard)]
    cta_meshes = [r.target.layout.mesh for r in reshards
                  if r.target.layout.mesh.topology.name == "cta"]
    thread_meshes = [r.target.layout.mesh for r in reshards
                     if r.target.layout.mesh.topology.name == "thread"]
    assert cta_meshes and thread_meshes
    assert cta_meshes[0].layout.shape == (128,)
    assert thread_meshes[0].layout.shape == (8, 32)
