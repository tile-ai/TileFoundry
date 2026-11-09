"""CUDA MeshScope lowering and its layout helpers."""

from __future__ import annotations

from tilefoundry import codegen, target
from tilefoundry.ir import core, tir, types
from tilefoundry_nvidia.cuda import target as cuda_target
from tilefoundry_nvidia.cuda.codegen import context
from tilefoundry_nvidia.cuda.codegen.tir import cute_layout


def resolved(topology: types.Topology | str) -> types.Topology:
    """*topology* as a ``Topology``, never a bare level name."""
    if isinstance(topology, str):
        raise RuntimeError(
            f"CUDA mesh emission: mesh level {topology!r} is still a bare name; "
            f"lowering resolves each level to a Topology stating its extent"
        )
    return topology


def program_topologies(mesh: types.Mesh) -> tuple[types.Topology, ...]:
    return tuple(resolved(t) for t in mesh.topologies)


def validate_topology(mesh: types.Mesh, tgt) -> None:
    """Validate that the target supports every program topology level.

    Each program topology level a mesh binds must be one this target
    supports; finer levels (e.g. warp) belong in the mesh layout, not as a
    program topology level. Defense-in-depth alongside the declared-topology
    check at lowering entry.
    """
    validate_cuda_topology_levels(tgt, (resolved(t).name for t in mesh.topologies))


def levelwise_layout(mesh: types.Mesh) -> str:
    """One nest per topology level, each in that level's own numbering.

    A mesh states one arrangement per level already, each in that level's own
    numbering, so a nest is that arrangement written out: nothing here decides
    where a boundary falls. The composite is a grouping, not a map: only the
    nests are evaluated.
    """
    arrangements = types.flattened_levels(mesh)
    return cute_layout.render_layout_type(
        tuple(shape for shape, _ in arrangements),
        tuple(strides for _, strides in arrangements),
    )


def mesh_type(mesh: types.Mesh) -> str:
    """The C++ ``tilefoundry::Mesh`` type for *mesh*, offset included."""
    topos = program_topologies(mesh)
    layout_value = mesh.layout
    if isinstance(layout_value, types.ComposedLayout):
        outer = layout_value.outer
        if not isinstance(outer, types.Layout):
            raise NotImplementedError(
                "CUDA mesh emission: a sliced mesh needs its participating box "
                "as a strided Layout; this states an identity box, with no sub-box"
            )
        base = tir.sync.participation(mesh).base
    else:
        base = 0
    if len(topos) == 1:
        shape, strides = types.flattened_levels(mesh)[0]
        layout = cute_layout.render_layout_type(shape, strides)
    else:
        if base:
            raise NotImplementedError(
                "CUDA mesh emission: a mesh naming several levels cannot also "
                "be sliced; the slice and the level boundary would both be "
                "deciding which positions these are"
            )
        layout = levelwise_layout(mesh)
    if base:
        layout = f"cute::ComposedLayout<cute::identity, cute::Int<{base}>, {layout}>"
    return f"tilefoundry::Mesh<{layout}, {', '.join(context.topology_scope_str(t.name) for t in topos)}>"


def is_dynamic_mesh(mesh: types.Mesh) -> bool:
    """A launch-provided (dynamic) CTA mesh.

    A launch-provided (dynamic) CTA mesh: its topology size or a layout axis
    extent is ``None`` and only known at launch time.
    """
    if any(t.size is None for t in program_topologies(mesh)):
        return True
    return any(s is None for s in types.flatten(mesh.layout).shape)


def validate_cuda_topology_levels(tgt: target.Target, names) -> None:
    """Validate names at the CUDA lowering boundary.

    See docs/spec/target.md § Topology levels.
    """
    supported = tuple(
        level.name for level in tgt.get_facts(target.TopologyFacts).topologies
    )
    for name in names:
        if name not in supported:
            raise ValueError(
                f"cuda target supports {{{', '.join(supported)}}} "
                f"topology levels; got {name!r}"
            )


@codegen.register(cuda_target.Target, codegen.Role.EMIT, core.MeshScope)
def emit_mesh_scope(node: core.MeshScope, emitter) -> None:
    ctx = emitter.context
    if ctx.target is None:
        raise RuntimeError("CUDA MeshScope emission requires its Target")
    validate_topology(node.mesh, ctx.target)
    name = ctx.name_for(node.binding)
    ctx.emit(f"// mesh scope: {program_topologies(node.mesh)[0].name}")

    is_slice = isinstance(node.mesh.layout, types.ComposedLayout)
    ctx.emit("{")
    ctx.indent()
    with ctx.nested_mesh_aliases():
        if not is_dynamic_mesh(node.mesh):
            alias = f"{name}_mesh_t"
            mesh_type_str = mesh_type(node.mesh)
            ctx.alias_mesh(node.mesh, alias, mesh_type_str)
            ctx.emit(f"using {alias} = {mesh_type_str};")
            ctx.emit(f"constexpr {alias} {name}_mesh{{}};")
        if is_slice:
            ctx.emit(f"if (tilefoundry::contains({name}_mesh, tilefoundry::program_ids())) {{")
            ctx.indent()
        emitter.visit(node.body)
        if is_slice:
            ctx.dedent()
            ctx.emit("}")
    ctx.dedent()
    ctx.emit("}")


__all__ = ["emit_mesh_scope", "mesh_type", "program_topologies", "validate_cuda_topology_levels"]
