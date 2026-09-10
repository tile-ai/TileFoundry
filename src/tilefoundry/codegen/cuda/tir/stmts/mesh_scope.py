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
from tilefoundry.ir.types.shard.mesh import Mesh, Topology, positions_at
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
    validate_cuda_topology_levels(target, (_resolved(t).name for t in mesh.topologies))


def _levelwise_layout(mesh: Mesh, topos) -> str:
    """One nest per topology level, each in that level's own numbering.

    A flat shape says nothing about which axes are whose, so the runtime's
    ``get<level>`` needs the boundary stated. ``level_axes`` hands the axes to
    the levels left to right and ``positions_at`` divides each level's strides
    by what the levels under it contribute, so a nest reads as the layout that
    level would have alone. The composite is a grouping, not a map: only the
    nests are evaluated. A level every instance shares keeps a mode of one, so
    that ``get<level>`` still has something to pick.
    """
    shapes, strides = [], []
    for topology in topos:
        level_shape, level_strides = positions_at(mesh, topology.name)
        if not level_shape:
            level_shape, level_strides = (1,), (0,)
        shapes.append(
            "cute::Shape<"
            + ", ".join(f"cute::Int<{s}>" for s in level_shape)
            + ">"
        )
        strides.append(
            "cute::Stride<"
            + ", ".join(f"cute::Int<{s}>" for s in level_strides)
            + ">"
        )
    return (
        f"cute::Layout<cute::Shape<{', '.join(shapes)}>, "
        f"cute::Stride<{', '.join(strides)}>>"
    )


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
    layout_value = mesh.layout
    if isinstance(layout_value, ComposedLayout):
        outer = layout_value.outer
        if not isinstance(outer, Layout) or outer.strides is None:
            raise NotImplementedError(
                "CUDA mesh emission: a sliced mesh needs its participating box "
                "as a strided Layout; this states an identity box, with no sub-box"
            )
        shape, strides, base = outer.shape, outer.strides, participation(mesh).base
    else:
        if layout_value.strides is None:
            raise NotImplementedError("CUDA mesh emission: mesh layout needs strides")
        shape, strides, base = layout_value.shape, layout_value.strides, 0
    if len(topos) == 1:
        shape_types = ", ".join(f"cute::Int<{s}>" for s in shape)
        stride_types = ", ".join(f"cute::Int<{s}>" for s in strides)
        layout = (
            f"cute::Layout<cute::Shape<{shape_types}>, "
            f"cute::Stride<{stride_types}>>"
        )
    else:
        if base:
            raise NotImplementedError(
                "CUDA mesh emission: a mesh naming several levels cannot also "
                "be sliced; the slice and the level boundary would both be "
                "deciding which positions these are"
            )
        layout = _levelwise_layout(mesh, topos)
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
    return any(s is None for s in mesh.layout.shape)


@register_codegen_cuda(MeshScope)
def _emit(node: MeshScope, ctx: CodegenContext) -> None:
    """Emit the block a mesh scope is, and the mesh object it states.

    The scope states the mesh once, as the object everything under it reads: a
    view's shard layout names this one rather than rebuilding an equal mesh of
    its own beside it. Both names are the C++ block's, so they leave with it --
    another kernel's scope states an equal mesh under a name this one cannot
    see, which is why the alias table is saved and restored around the body.
    """
    if ctx.target is None:
        raise RuntimeError("CUDA MeshScope emission requires its Target")
    _validate_topology(node.mesh, ctx.target)
    name = ctx.name_for(node.binding)
    ctx.emit(f"// mesh scope: {program_topologies(node.mesh)[0].name}")

    is_slice = isinstance(node.mesh.layout, ComposedLayout)
    ctx.emit("{")
    ctx.indent()
    outer_aliases = ctx._mesh_aliases
    ctx._mesh_aliases = dict(outer_aliases)
    try:
        if not _is_dynamic_mesh(node.mesh):
            alias = f"{name}_mesh_t"
            mesh_type_str = mesh_type(node.mesh)
            ctx._mesh_aliases[id(node.mesh)] = (alias, mesh_type_str)
            ctx.emit(f"using {alias} = {mesh_type_str};")
            ctx.emit(f"constexpr {alias} {name}_mesh{{}};")
        if is_slice:
            ctx.emit(
                f"if (tilefoundry::contains({name}_mesh, "
                "tilefoundry::program_ids())) {"
            )
            ctx.indent()
        ctx.emit_node(node.body)
        if is_slice:
            ctx.dedent()
            ctx.emit("}")
    finally:
        ctx._mesh_aliases = outer_aliases
    ctx.dedent()
    ctx.emit("}")
