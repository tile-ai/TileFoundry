"""Construction and specialization helpers for IR patterns."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from enum import Enum

from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.types import (
    DType,
    StorageKind,
)

from . import predicates as P
from .match import PatternMatcher, evaluated
from .pattern import (
    AndPattern,
    ComposedLayoutPattern,
    LayoutPattern,
    MeshPattern,
    OrPattern,
    Pattern,
    RangePattern,
    ShardLayoutPattern,
    StarPattern,
    SwitchPattern,
    TensorPattern,
    WildcardPattern,
)

MOVED_STORAGES = (StorageKind.GMEM, StorageKind.SMEM, StorageKind.RMEM)


def tensor_in(
    storage: StorageKind,
    execution_mesh: MeshPattern | None = None,
) -> TensorPattern:
    """A tensor of any shape and dtype held in *storage*."""
    layout = None
    if storage == StorageKind.RMEM and execution_mesh is not None:
        layout = _sharded_on(execution_mesh)
    return TensorPattern(storage=storage, layout=layout)


def operand_tile(
    index: int,
    storage=None,
    layout=None,
    execution_mesh: MeshPattern | None = None,
) -> TensorPattern:
    """A tensor tile whose dtype and storage captures are named by operand slot."""
    storages = MOVED_STORAGES if storage is None else storage
    dtype_name = dtype_place(index)
    storage_name = storage_place(index)
    predicates = [P.Bits(dtype_name) % 8 == 0]
    if isinstance(storages, tuple):
        predicates.append(P.In(WildcardPattern(storage_name), storages))
    if layout is None and execution_mesh is not None:
        sharded = _sharded_on(execution_mesh)
        if isinstance(storages, tuple):
            layout = SwitchPattern(
                storage_name,
                {
                    held: sharded if held == StorageKind.RMEM else WildcardPattern()
                    for held in storages
                },
            )
        elif storages == StorageKind.RMEM:
            layout = sharded
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


_THREAD_EXECUTION_LAYOUT = ComposedLayoutPattern(
    inner=None,
    offset=WildcardPattern("p0"),
    outer=LayoutPattern(
        (StarPattern(WildcardPattern("execution_shape")),),
        (StarPattern(WildcardPattern("execution_strides")),),
    ),
)


def thread_execution_mesh() -> MeshPattern:
    """Declare one contiguous non-empty run of executing threads."""
    return MeshPattern(("thread",), _THREAD_EXECUTION_LAYOUT)


def memory_execution_mesh() -> MeshPattern:
    """Bind a contiguous run at the current innermost topology level."""
    return MeshPattern(WildcardPattern("execution_topologies"), _THREAD_EXECUTION_LAYOUT)


def declared_execution_mesh(op_type) -> MeshPattern | OrPattern:
    """Return one op's reserved execution-mesh declaration or reject it."""
    declaration = op_type if isinstance(op_type, type) else type(op_type)
    factory = getattr(op_type, "execution_mesh_pattern", None)
    if callable(factory):
        declared = factory()
    else:
        stated = getattr(declaration, "execution_mesh", None)
        declared = stated.pattern if isinstance(stated, ParamDef) else stated
    choices = declared.patterns if isinstance(declared, OrPattern) else (declared,)
    if not choices or not all(isinstance(choice, MeshPattern) for choice in choices):
        name = getattr(declaration, "reference_name", "") or declaration.__name__
        raise ValueError(
            f"{name} execution_mesh must be a constant, ParamDef, or operand mesh reference"
        )
    return declared


def _sharded_on(execution_mesh: MeshPattern) -> OrPattern:
    """A lexical tile, or a shard explicitly held by *execution_mesh*."""
    return OrPattern(
        ShardLayoutPattern(
            layout=WildcardPattern(),
            attrs=WildcardPattern(),
            mesh=execution_mesh,
        ),
        LayoutPattern(),
    )


def selected_pattern(pattern, bindings: dict):
    """Resolve declaration branches fixed by already-bound attributes."""
    if isinstance(pattern, TensorPattern):
        return replace(pattern, layout=selected_pattern(pattern.layout, bindings))
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


def variants(
    parameters: tuple[ParamDef, ...],
    *,
    vary_defaulted: bool,
    values: Callable[[ParamDef], tuple] | None = None,
) -> tuple[dict, ...]:
    """Every binding of *parameters* that their patterns admit, in declaration order.

    Enum parameters take their members in declaration order and DType parameters
    every declared dtype; ``values`` supplies the others, which stay unbound
    without it or when it answers ``None``, while an empty answer admits no
    binding. A parameter with a default is varied only with ``vary_defaulted``.
    No parameters give one empty binding.
    """
    states: tuple[dict, ...] = ({},)
    for param in parameters:
        if param.has_default and not vary_defaulted:
            continue
        options = _finite_options(param, values)
        if options is None:
            continue
        held = []
        for state in states:
            for value in options:
                matcher = PatternMatcher(state)
                if matcher.match(param.pattern, value) and matcher.solve():
                    held.append({**state, param.name: value})
        states = tuple(held)
    return states


def _finite_options(param: ParamDef, values) -> tuple | None:
    annotation = param.annotation
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return tuple(annotation)
    if annotation is DType:
        return tuple(DType._members().values())
    return None if values is None else values(param)


def fixed_pattern_value(value, bindings: dict):
    """Resolve one declaration value when every symbolic leaf is bound."""
    if isinstance(value, SwitchPattern):
        if value.param not in bindings:
            return None
        return fixed_pattern_value(dict(value.branches).get(bindings[value.param]), bindings)
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


def declared_symbol_axes(
    params: tuple, op, bindings: dict, name: str
) -> tuple[tuple[int, int], ...]:
    """Operand axes whose declared extent is the symbol named *name*."""
    axes = []
    for index, param in enumerate(params):
        pattern = param.pattern.read_on(op) if hasattr(param.pattern, "read_on") else param.pattern
        pattern = selected_pattern(pattern, bindings)
        for axis, dim in enumerate(getattr(pattern, "shape", None) or ()):
            if getattr(dim, "name", None) == name:
                axes.append((index, axis))
    return tuple(axes)


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
    "declared_execution_mesh",
    "dtype_place",
    "variants",
    "locate_dim_var",
    "operand_tile",
    "storage_place",
    "tensor_in",
    "thread_execution_mesh",
    "whole_vectors",
]
