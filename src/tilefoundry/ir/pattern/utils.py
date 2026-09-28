"""Construction and specialization helpers for IR patterns."""

from __future__ import annotations

from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.types import (
    ComposedLayout,
    Layout,
    Mesh,
    ShardLayout,
    StorageKind,
    TensorType,
)
from tilefoundry.ir.types.mesh import separate, starts
from tilefoundry.ir.types.stride import compact_row_major

from . import predicates as P
from .constraint import DistinctConstraint
from .match import between_rules, evaluated
from .pattern import (
    AndPattern,
    ComposedLayoutPattern,
    LayoutPattern,
    MeshPattern,
    OrPattern,
    Pattern,
    RangePattern,
    ShardLayoutPattern,
    SwitchPattern,
    TensorPattern,
    WildcardPattern,
)

MOVED_STORAGES = (StorageKind.GMEM, StorageKind.SMEM, StorageKind.RMEM)


def tensor_in(storage: StorageKind) -> TensorPattern:
    """A tensor of any shape and dtype held in *storage*."""
    return TensorPattern(storage=storage)


def operand_tile(index: int, storage=None, layout=None) -> TensorPattern:
    """A tensor tile whose dtype and storage captures are named by operand slot."""
    storages = MOVED_STORAGES if storage is None else storage
    dtype_name = dtype_place(index)
    storage_name = storage_place(index)
    predicates = [P.Bits(dtype_name) % 8 == 0]
    if isinstance(storages, tuple):
        predicates.append(P.In(WildcardPattern(storage_name), storages))
    return TensorPattern(
        dtype=WildcardPattern(dtype_name),
        storage=(WildcardPattern(storage_name) if isinstance(storages, tuple) else storages),
        layout=layout,
        predicates=tuple(predicates),
    )


def dtype_place(index: int) -> str:
    """The capture name for transfer end *index*'s dtype."""
    return f"dtype{index}"


def storage_place(index: int) -> str:
    """The capture name for transfer end *index*'s storage."""
    return f"storage{index}"


def whole_vectors(index: int, widths: tuple[int, ...]) -> LayoutPattern:
    """Whole vectors at *widths*, counted in operand *index*'s element dtype."""
    return LayoutPattern(
        predicates=(
            P.WholeVectors(
                WildcardPattern("width"),
                dtype_place(index),
                widths,
            ),
        )
    )


_ANY_THREADS = ComposedLayoutPattern(
    inner=None,
    offset=WildcardPattern(),
    outer=LayoutPattern(
        ((WildcardPattern("n"),),),
        ((1,),),
        predicates=(
            WildcardPattern("n") >= 1,
            P.Forward(per_mode=True),
            P.Injective(per_mode=True),
        ),
    ),
)


def any_threads() -> ParamDef:
    """Declare an optional scope spanning one or more threads."""
    return ParamDef(
        kind="attribute",
        annotation=Mesh,
        pattern=MeshPattern(("thread",), _ANY_THREADS),
        optional=True,
        default=None,
    )


def selected_pattern(pattern, bindings: dict):
    """Resolve declaration branches fixed by already-bound attributes."""
    if isinstance(pattern, SwitchPattern) and pattern.param in bindings:
        branch = dict(pattern.branches).get(bindings[pattern.param])
        return None if branch is None else selected_pattern(branch, bindings)
    if isinstance(pattern, (AndPattern, OrPattern)):
        parts = pattern.parts if isinstance(pattern, AndPattern) else pattern.patterns
        selected = tuple(
            held for part in parts if (held := selected_pattern(part, bindings)) is not None
        )
        if len(selected) == 1:
            return selected[0]
    return pattern


def fixed_pattern_value(value, bindings: dict):
    """Resolve one declaration value when every symbolic leaf is bound."""
    if isinstance(value, tuple):
        parts = tuple(fixed_pattern_value(item, bindings) for item in value)
        return None if any(item is None for item in parts) else parts
    if value is None or type(value) in (int, str):
        return value
    name = getattr(value, "name", None)
    if name in bindings:
        return bindings[name]
    resolved = evaluated(value, bindings)
    return resolved if resolved is not None else value if name is None else None


def declared_shape(pattern: TensorPattern | None, bindings: dict) -> tuple | None:
    """Return a tensor declaration's concrete shape, when it states one."""
    if pattern is None or pattern.shape is None:
        return None
    shape = fixed_pattern_value(pattern.shape, bindings)
    return shape if isinstance(shape, tuple) and all(type(dim) is int for dim in shape) else None


def declared_layout(pattern, bindings: dict, mesh: Mesh | None):
    """Materialize a concrete write-only layout from its declaration."""
    pattern = selected_pattern(pattern, bindings)
    if isinstance(pattern, LayoutPattern):
        shape = fixed_pattern_value(pattern.shape, bindings)
        strides = fixed_pattern_value(pattern.strides, bindings)
        return None if shape is None or strides is None else Layout(shape, strides)
    if isinstance(pattern, ComposedLayoutPattern):
        inner = fixed_pattern_value(pattern.inner, bindings)
        offset = fixed_pattern_value(pattern.offset, bindings)
        outer = declared_layout(pattern.outer, bindings, mesh)
        return None if offset is None or outer is None else ComposedLayout(inner, offset, outer)
    if not isinstance(pattern, ShardLayoutPattern):
        return None
    inner = declared_layout(pattern.layout, bindings, mesh)
    frame = (
        next(
            (
                level
                for level in separate(mesh)
                if getattr(level.topologies[0], "name", level.topologies[0]) == "thread"
            ),
            None,
        )
        if mesh is not None
        else None
    )
    if frame is not None:
        mesh_layout = declared_layout(
            pattern.mesh.layout,
            {**bindings, "p0": starts(frame)[0]},
            mesh,
        )
        if mesh_layout is not None:
            frame = Mesh(frame.topologies, mesh_layout, frame.names)
    return None if inner is None or frame is None else ShardLayout(inner, pattern.attrs, frame)


def declared_write_type(op, param, pattern: TensorPattern | None, inputs: dict, mesh) -> TensorType:
    """Resolve one write-only operand from its declaration and read operands."""
    if pattern is None:
        raise ValueError(
            f"{type(op).__name__} {param.name} is write-only and declares no result shape"
        )
    layout_storages = {
        "gmem_layout": StorageKind.GMEM,
        "smem_layout": StorageKind.SMEM,
        "rmem_layout": StorageKind.RMEM,
    }
    stated = tuple(
        (storage, getattr(op, name))
        for name, storage in layout_storages.items()
        if getattr(op, name, None) is not None
    )
    source = next(iter(inputs.values()))
    bindings = dict(getattr(getattr(op, "atom", None), "bindings", {}))
    shape = declared_shape(pattern, bindings) or tuple(source.shape)
    dtype = pattern.dtype if hasattr(pattern.dtype, "bit_width") else source.dtype
    layout = None
    if isinstance(pattern.storage, StorageKind):
        storage = pattern.storage
        layout = next((held_layout for held, held_layout in stated if held is storage), None)
    elif len(stated) == 1:
        storage, layout = stated[0]
    else:
        storage = None
        for rule in between_rules(type(op)):
            if not isinstance(rule, DistinctConstraint) or rule.field != "storage":
                continue
            if param.name not in (rule.left, rule.right):
                continue
            other = rule.right if param.name == rule.left else rule.left
            if other in inputs:
                choices = tuple(
                    held
                    for held in (StorageKind.GMEM, StorageKind.SMEM)
                    if held != inputs[other].storage
                )
                storage = choices[0] if len(choices) == 1 else None
                break
        if storage is None:
            raise ValueError(f"{type(op).__name__} does not determine {param.name} storage")
    layout = layout or declared_layout(pattern.layout, bindings, mesh)
    layout = layout or Layout(shape, tuple(compact_row_major(shape)))
    return TensorType(shape, dtype, layout, storage)


def locate_dim_var(params: tuple, name: str) -> tuple[int, int] | None:
    """Return the first parameter/axis carrying a ``DimVar`` named *name*."""
    for index, param in enumerate(params):
        shape = getattr(param.type, "shape", None)
        if shape is None:
            continue
        for axis, dim in enumerate(shape):
            if getattr(dim, "name", None) == name:
                return index, axis
    return None


def _mangle_variant_name(name: str, specializations: tuple[Pattern, ...]) -> str:
    if len(specializations) != 1 or not isinstance(specializations[0], RangePattern):
        raise TypeError("variant requires exactly one RangePattern")
    pattern = specializations[0]
    if not pattern.dim_var or pattern.lo is None or pattern.hi is None:
        raise TypeError("variant RangePattern requires dim_var, lo, and hi")
    return f"{name}${pattern.dim_var}${pattern.lo}_{pattern.hi}"


__all__ = [
    "MOVED_STORAGES",
    "_mangle_variant_name",
    "any_threads",
    "dtype_place",
    "locate_dim_var",
    "operand_tile",
    "storage_place",
    "tensor_in",
    "whole_vectors",
]
