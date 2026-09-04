"""Shared Python rendering primitives used by HIR and TIR printers."""

from __future__ import annotations


def _printer():
    """Load the legacy printer lazily so its import graph stays acyclic."""
    from . import python_printer  # noqa: PLC0415

    return python_printer


def layout_str(layout, indent: str = "") -> str:
    return _printer()._layout_str(layout, indent)


def mesh_str(mesh, indent: str = "") -> str:
    return _printer()._mesh_str(mesh, indent)


def shard_layout_str(layout, indent: str = "", *, mesh_ref: str | None = None) -> str:
    return _printer()._shard_layout_str(layout, indent, mesh_ref=mesh_ref)


def tensor_annotation(ty, *, mesh_name_map=None, indent: str = "", is_const: bool = False) -> str:
    return _printer()._tensor_annotation(
        ty, mesh_name_map=mesh_name_map, indent=indent, is_const=is_const
    )


def dtype_str(dtype) -> str:
    return _printer()._dtype_str(dtype)


def shape_tuple(shape) -> str:
    return _printer()._shape_tuple(shape)


def mesh_name_map(meshes):
    return _printer()._mesh_name_map(meshes)


def pattern_ctor(pattern) -> str:
    return _printer()._pattern_ctor(pattern)


__all__ = [
    "layout_str",
    "mesh_str",
    "shard_layout_str",
    "tensor_annotation",
    "dtype_str",
    "shape_tuple",
    "mesh_name_map",
    "pattern_ctor",
]
