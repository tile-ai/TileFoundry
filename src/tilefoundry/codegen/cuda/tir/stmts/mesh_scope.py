"""Emitter for `tir.MeshScope` — emits a C++ block + comment marker +
constexpr Mesh type alias (spec 010 §5)."""

from __future__ import annotations

from tilefoundry.codegen.cuda.context import (
    CodegenContext,
    register_codegen_cuda,
    topology_scope_str,
)
from tilefoundry.ir.tir.stmts import MeshScope
from tilefoundry.target import validate_cuda_topology_levels


def _validate_topology(mesh) -> None:
    """Each program topology level a mesh binds must be one this target
    supports; finer levels (e.g. warp) belong in the mesh layout, not as a
    program topology level. Defense-in-depth alongside the declared-topology
    check at lowering entry."""
    validate_cuda_topology_levels(t.name for t in mesh.topologies)




def _mesh_type(mesh) -> str:
    topo = mesh.topology
    shape_types = ", ".join(f"cute::Int<{s}>" for s in mesh.layout.shape)
    stride_types = ", ".join(f"cute::Int<{s}>" for s in mesh.layout.strides)
    return (
        f"tilefoundry::Mesh<"
        f"tilefoundry::Topology<{topology_scope_str(topo.name)}, {topo.size}>, "
        f"cute::Layout<cute::Shape<{shape_types}>, cute::Stride<{stride_types}>>>"
    )


def _is_dynamic_mesh(mesh) -> bool:
    """A launch-provided (dynamic) CTA mesh: its topology size or a layout axis
    extent is ``None`` and only known at launch time."""
    if mesh.topology.size is None:
        return True
    return any(s is None for s in mesh.layout.shape)


@register_codegen_cuda(MeshScope)
def _emit(node: MeshScope, ctx: CodegenContext) -> None:
    _validate_topology(node.mesh)
    name = ctx.name_for(node.binding)
    ctx.emit(f"// mesh scope: {node.mesh.topology.name}")
    # A dynamic mesh has no compile-time type: its extent comes from the launch
    # grid, so no constexpr ``using`` alias is emitted. Shard layouts on a
    # dynamic mesh are built as runtime values at their use sites (reshard /
    # make_shard_tensor) rather than referencing the alias.
    if not _is_dynamic_mesh(node.mesh):
        alias = f"{name}_mesh_t"
        mesh_type_str = _mesh_type(node.mesh)
        ctx._mesh_aliases[id(node.mesh)] = (alias, mesh_type_str)
        ctx.emit(f"using {alias} = {mesh_type_str};")
    ctx.emit("{")
    ctx.indent()
    ctx.emit_node(node.body)
    ctx.dedent()
    ctx.emit("}")
