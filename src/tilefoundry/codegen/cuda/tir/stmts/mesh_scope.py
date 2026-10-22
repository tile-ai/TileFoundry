"""Helpers for CUDA ``MeshScope`` lowering.

The ``CudaEmitter`` owns Stmt traversal and invokes these stateless mesh
layout helpers while visiting a ``MeshScope``.
"""

from __future__ import annotations

from tilefoundry.codegen.cuda.context import topology_scope_str
from tilefoundry.codegen.cuda.tir.cute_layout import render_layout_type
from tilefoundry.ir.tir.sync import participation
from tilefoundry.ir.types.layout import ComposedLayout, Layout, flatten
from tilefoundry.ir.types.mesh import Mesh, Topology, flattened_levels
from tilefoundry.target import validate_cuda_topology_levels


def _resolved(topology: Topology | str) -> Topology:
    """*topology* as a ``Topology``, never a bare level name."""
    if isinstance(topology, str):
        raise RuntimeError(
            f"CUDA mesh emission: mesh level {topology!r} is still a bare name; "
            f"lowering resolves each level to a Topology stating its extent"
        )
    return topology


def program_topologies(mesh: Mesh) -> tuple[Topology, ...]:
    return tuple(_resolved(t) for t in mesh.topologies)


def _validate_topology(mesh: Mesh, target) -> None:
    """Validate that the target supports every program topology level.

    Each program topology level a mesh binds must be one this target
    supports; finer levels (e.g. warp) belong in the mesh layout, not as a
    program topology level. Defense-in-depth alongside the declared-topology
    check at lowering entry.
    """
    validate_cuda_topology_levels(target, (_resolved(t).name for t in mesh.topologies))


def _levelwise_layout(mesh: Mesh) -> str:
    """One nest per topology level, each in that level's own numbering.

    A mesh states one arrangement per level already, each in that level's own
    numbering, so a nest is that arrangement written out: nothing here decides
    where a boundary falls. The composite is a grouping, not a map: only the
    nests are evaluated.
    """
    arrangements = flattened_levels(mesh)
    return render_layout_type(
        tuple(shape for shape, _ in arrangements),
        tuple(strides for _, strides in arrangements),
    )


def mesh_type(mesh: Mesh) -> str:
    """The C++ ``tilefoundry::Mesh`` type for *mesh*, offset included."""
    topos = program_topologies(mesh)
    layout_value = mesh.layout
    if isinstance(layout_value, ComposedLayout):
        outer = layout_value.outer
        if not isinstance(outer, Layout):
            raise NotImplementedError(
                "CUDA mesh emission: a sliced mesh needs its participating box "
                "as a strided Layout; this states an identity box, with no sub-box"
            )
        base = participation(mesh).base
    else:
        base = 0
    if len(topos) == 1:
        shape, strides = flattened_levels(mesh)[0]
        layout = render_layout_type(shape, strides)
    else:
        if base:
            raise NotImplementedError(
                "CUDA mesh emission: a mesh naming several levels cannot also "
                "be sliced; the slice and the level boundary would both be "
                "deciding which positions these are"
            )
        layout = _levelwise_layout(mesh)
    if base:
        layout = f"cute::ComposedLayout<cute::identity, cute::Int<{base}>, {layout}>"
    return f"tilefoundry::Mesh<{layout}, {', '.join(topology_scope_str(t.name) for t in topos)}>"


def _is_dynamic_mesh(mesh: Mesh) -> bool:
    """A launch-provided (dynamic) CTA mesh.

    A launch-provided (dynamic) CTA mesh: its topology size or a layout axis
    extent is ``None`` and only known at launch time.
    """
    if any(t.size is None for t in program_topologies(mesh)):
        return True
    return any(s is None for s in flatten(mesh.layout).shape)


__all__ = ["mesh_type", "program_topologies"]
