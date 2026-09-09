"""Emit CUDA mesh scopes.

Emitter for `tir.MeshScope` — emits a C++ block + comment marker +
constexpr Mesh type alias ([runtime §2.3](docs/spec/runtime.md#23-tilefoundrymesh)).
"""

from __future__ import annotations

from tilefoundry.codegen.cuda.context import (
    CodegenContext,
    register_codegen_cuda,
    topology_scope_str,
)
from tilefoundry.ir.tir.stmts import MeshScope
from tilefoundry.ir.tir.sync import participation
from tilefoundry.ir.types.shard.layout import ComposedLayout, Layout
from tilefoundry.ir.types.shard.mesh import Mesh, Topology
from tilefoundry.target import validate_cuda_topology_levels


def _resolved(topology: Topology | str) -> Topology:
    """*topology* as a ``Topology``, never a bare level name.

    ``Mesh.topologies`` admits a name because the authored surface writes one
    (``Mesh(("cta",), ...)``) and the parser resolves it against the module's
    declaration before lowering. One still spelled as a string here never got
    that resolution, so it states no extent, and codegen has nowhere else to
    read one from.
    """
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
    validate_cuda_topology_levels(
        target, (_resolved(t).name for t in mesh.topologies)
    )


def mesh_geometry(mesh: Mesh) -> tuple[tuple, tuple, int]:
    """The shape, the strides, and the first instance a C++ ``Mesh`` covers.

    A sliced mesh keeps the participating sub-box in ``ComposedLayout.outer``
    and where that box starts in the composed offset, so the two come apart
    here; an un-sliced mesh is the whole level and starts at zero. The base
    comes from ``participation`` rather than the offset field so a slice that
    is not one contiguous run of instances is refused, not emitted.
    """
    layout = mesh.layout
    if isinstance(layout, ComposedLayout):
        outer = layout.outer
        if not isinstance(outer, Layout) or outer.strides is None:
            raise NotImplementedError(
                f"CUDA mesh emission: a sliced mesh needs its participating box "
                f"as a strided Layout in ComposedLayout.outer; {outer!r} states "
                f"an identity box, whose extents are the inner component's and "
                f"so name no sub-box for the offset to start"
            )
        return outer.shape, outer.strides, participation(mesh).base
    if layout.strides is None:
        raise NotImplementedError(
            "CUDA mesh emission: a mesh layout must state its strides; "
            "None leaves which instance owns which position unsaid"
        )
    return layout.shape, layout.strides, 0


def mesh_type(mesh: Mesh) -> str:
    """The C++ ``tilefoundry::Mesh`` type for *mesh*, offset included.

    A slice's origin rides in the layout, not beside it: ``ComposedLayout(inner,
    offset, outer)`` here is ``cute::ComposedLayout<cute::identity,
    cute::Int<offset>, cute::Layout<...>>``, whose ``operator()`` is
    ``offset + outer(c)`` -- the same map the IR layout makes. It has to be
    carried: without it every slice would look like the block it came from, and
    ``ops::sync`` reads it to tell the two apart.
    """
    topos = program_topologies(mesh)
    shape, strides, base = mesh_geometry(mesh)
    shape_types = ", ".join(f"cute::Int<{s}>" for s in shape)
    stride_types = ", ".join(f"cute::Int<{s}>" for s in strides)
    layout = (
        f"cute::Layout<cute::Shape<{shape_types}>, cute::Stride<{stride_types}>>"
    )
    if base:
        layout = (
            f"cute::ComposedLayout<cute::identity, cute::Int<{base}>, {layout}>"
        )
    return (
        f"tilefoundry::Mesh<"
        f"{layout}, "
        f"{', '.join(topology_scope_str(t.name) for t in topos)}>"
    )


def _is_dynamic_mesh(mesh: Mesh) -> bool:
    """A launch-provided (dynamic) CTA mesh.

    A launch-provided (dynamic) CTA mesh: its topology size or a layout axis
    extent is ``None`` and only known at launch time.
    """
    if any(t.size is None for t in program_topologies(mesh)):
        return True
    return any(s is None for s in mesh.layout.shape)


@register_codegen_cuda(MeshScope)
def _emit(node: MeshScope, ctx: CodegenContext) -> None:
    if ctx.target is None:
        raise RuntimeError("CUDA MeshScope emission requires its Target")
    _validate_topology(node.mesh, ctx.target)
    name = ctx.name_for(node.binding)
    ctx.emit(f"// mesh scope: {program_topologies(node.mesh)[0].name}")




    if not _is_dynamic_mesh(node.mesh):
        alias = f"{name}_mesh_t"
        mesh_type_str = mesh_type(node.mesh)
        ctx._mesh_aliases[id(node.mesh)] = (alias, mesh_type_str)
        ctx.emit(f"using {alias} = {mesh_type_str};")
    ctx.emit("{")
    ctx.indent()
    ctx.emit_node(node.body)
    ctx.dedent()
    ctx.emit("}")
