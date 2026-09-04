"""Python renderers shared by the HIR and TIR printers."""

from __future__ import annotations

from collections.abc import Callable

from tilefoundry.ir.core.pattern import DimVarRangePat, Pattern
from tilefoundry.ir.types import DType, TensorType
from tilefoundry.ir.types.shard.layout import ComposedLayout, Layout, LayoutBase
from tilefoundry.ir.types.shard.mesh import Mesh
from tilefoundry.ir.types.shard.shard_layout import Broadcast, Partial, ShardLayout, Split
from tilefoundry.ir.types.storage import StorageKind


def dtype_str(dtype: DType) -> str:
    return dtype.name


def shape_tuple(shape: tuple, render: Callable[[object], str] = str) -> str:
    values = tuple(render(entry) for entry in shape)
    return f"({values[0]},)" if len(values) == 1 else "(" + ", ".join(values) + ")"


def _shard_attr_str(attr) -> str:
    if isinstance(attr, Broadcast):
        return "B()"
    if isinstance(attr, Split):
        return f"S({attr.axis})"
    if isinstance(attr, Partial):
        return f'P("{attr.reduction}")'
    raise TypeError(f"unsupported shard attribute: {type(attr).__name__}")


def layout_str(layout: LayoutBase | None, indent: str = "", *, render=str) -> str:
    if layout is None:
        return "None"
    if isinstance(layout, Layout):
        strides = shape_tuple(layout.strides, render) if layout.strides is not None else "None"
        return f"Layout({shape_tuple(layout.shape, render)}, {strides})"
    if isinstance(layout, ShardLayout):
        return shard_layout_str(layout, indent, render=render)
    if isinstance(layout, ComposedLayout):
        child = indent + "    "
        return (
            "ComposedLayout(\n"
            f"{child}inner={layout_str(layout.inner, child, render=render)},\n"
            f"{child}offset={render(layout.offset)},\n"
            f"{child}outer={layout_str(layout.outer, child, render=render)},\n"
            f"{indent})"
        )
    raise TypeError(f"unsupported layout type: {type(layout).__name__}")


def topologies_str(mesh: Mesh, render=str) -> str:
    values = ", ".join(f'Topology("{t.name}", {render(t.size)})' for t in mesh.topologies)
    return f"({values}{',' if len(mesh.topologies) == 1 else ''})"


def mesh_str(mesh: Mesh, indent: str = "", *, render=str) -> str:
    result = f"Mesh({topologies_str(mesh, render)}, {layout_str(mesh.layout, indent, render=render)}"
    if mesh.names:
        result += f", names={tuple(mesh.names)!r}"
    return result + ")"


def shard_layout_str(layout: ShardLayout, indent: str = "", *, mesh_ref=None, render=str) -> str:
    child = indent + "    "
    attrs = ", ".join(_shard_attr_str(attr) for attr in layout.attrs)
    if len(layout.attrs) == 1:
        attrs += ","
    return (
        "ShardLayout(\n"
        f"{child}layout={layout_str(layout.layout, child, render=render)},\n"
        f"{child}attrs=({attrs}),\n"
        f"{child}mesh={mesh_ref or mesh_str(layout.mesh, child, render=render)},\n"
        f"{indent})"
    )


def tensor_annotation(ty: TensorType, *, mesh_name_map=None, indent="", is_const=False, render=str, surface=None) -> str:
    head = "ConstTensor" if is_const else "Tensor"
    result = f'{head}[{shape_tuple(ty.shape, render)}, "{dtype_str(ty.dtype)}"'
    if isinstance(ty.layout, ShardLayout):
        mesh_name = mesh_name_map.get(id(ty.layout.mesh)) if mesh_name_map else None
        sugar = surface(ty.layout, mesh_name=mesh_name, mesh_unique=len(mesh_name_map) == 1) if surface and mesh_name and ty.layout.mesh.names else None
        if sugar is not None:
            result += f", {sugar}"
        else:
            result += f",\n{indent}    {shard_layout_str(ty.layout, indent + '    ', render=render)}"
    if ty.storage is not StorageKind.GMEM:
        result += f', "{ty.storage.name.lower()}"'
    return result + "]"


def mesh_name_map(meshes: dict[int, Mesh]) -> dict[int, str]:
    used: set[str] = set()
    result = {}
    for identity, mesh in meshes.items():
        base = mesh.topologies[0].name if mesh.topologies else "mesh"
        name = base
        suffix = 2
        while name in used:
            name = f"{base}_{suffix}"
            suffix += 1
        used.add(name)
        result[identity] = name
    return result


def pattern_ctor(pattern: Pattern) -> str:
    if isinstance(pattern, DimVarRangePat):
        return f'DimVarRangePat("{pattern.dim_var}", {pattern.lo}, {pattern.hi})'
    return repr(pattern)


__all__ = ["dtype_str", "layout_str", "mesh_name_map", "mesh_str", "pattern_ctor", "shape_tuple", "shard_layout_str", "tensor_annotation", "topologies_str"]
