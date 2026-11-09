"""Construction and specialization helpers for IR patterns."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from enum import Enum

from tilefoundry import target
from tilefoundry.ir import expr

from .match import Matcher, Pattern, evaluated
from .pattern import (
    And,
    Bits,
    ComposedLayout,
    In,
    Layout,
    Mesh,
    Or,
    Range,
    ShardLayout,
    Star,
    Switch,
    Tensor,
    Wildcard,
)
from .predicates import WholeVectors

MOVED_STORAGES = (expr.StorageKind.GMEM, expr.StorageKind.SMEM, expr.StorageKind.RMEM)


def tensor_in(
    storage: expr.StorageKind,
    execution_mesh: Mesh | None = None,
) -> Tensor:
    """A tensor of any shape and dtype held in *storage*."""
    layout = None
    if storage == expr.StorageKind.RMEM and execution_mesh is not None:
        layout = sharded_on(execution_mesh)
    return Tensor(storage=storage, layout=layout)


def operand_tile(
    index: int,
    storage=None,
    layout=None,
    execution_mesh: Mesh | None = None,
) -> Tensor:
    """A tensor tile whose dtype and storage captures are named by operand slot."""
    storages = MOVED_STORAGES if storage is None else storage
    dtype_name = dtype_place(index)
    storage_name = storage_place(index)
    predicates = [Bits(dtype_name) % 8 == 0]
    if isinstance(storages, tuple):
        predicates.append(In(Wildcard(storage_name), storages))
    if layout is None and execution_mesh is not None:
        sharded = sharded_on(execution_mesh)
        if isinstance(storages, tuple):
            layout = Switch(
                storage_name,
                {
                    held: sharded if held == expr.StorageKind.RMEM else Wildcard()
                    for held in storages
                },
            )
        elif storages == expr.StorageKind.RMEM:
            layout = sharded
    return Tensor(
        dtype=Wildcard(dtype_name),
        storage=(Wildcard(storage_name) if isinstance(storages, tuple) else storages),
        layout=layout,
        predicates=tuple(predicates),
    )


def dtype_place(index: int) -> str:
    """The capture name for transfer end *index*'s dtype."""
    return f"dtype{index}"


def storage_place(index: int) -> str:
    """The capture name for transfer end *index*'s storage."""
    return f"storage{index}"


def whole_vectors(index: int, widths: tuple[int, ...]) -> Layout:
    """Whole vectors at *widths*, counted in operand *index*'s element dtype."""
    return Layout(
        predicates=(
            WholeVectors(
                Wildcard("width"),
                dtype_place(index),
                widths,
            ),
        )
    )


THREAD_EXECUTION_LAYOUT = ComposedLayout(
    inner=None,
    offset=Wildcard("p0"),
    outer=Layout(
        (Star(Wildcard("execution_shape")),),
        (Star(Wildcard("execution_strides")),),
    ),
)


def thread_execution_mesh() -> Mesh:
    """Declare one contiguous non-empty run of executing threads."""
    return Mesh(("thread",), THREAD_EXECUTION_LAYOUT)


def memory_execution_mesh() -> Mesh:
    """Bind a contiguous run at the current innermost topology level."""
    return Mesh(Wildcard(), THREAD_EXECUTION_LAYOUT)


def declared_execution_mesh(op_type, tgt=None) -> Mesh | Or:
    """Resolve one execution declaration, specializing memory levels from Target Facts."""
    declaration = op_type if isinstance(op_type, type) else type(op_type)
    factory = getattr(op_type, "execution_mesh_pattern", None)
    if callable(factory):
        declared = factory()
    else:
        stated = getattr(declaration, "execution_mesh", None)
        declared = stated.pattern if isinstance(stated, expr.ParamDef) else stated
    choices = declared.patterns if isinstance(declared, Or) else (declared,)
    if not choices or not all(isinstance(choice, Mesh) for choice in choices):
        name = expr.get_registered_name(declaration)
        raise ValueError(
            f"{name} execution_mesh must be a constant, ParamDef, or operand mesh reference"
        )
    if tgt is not None and any(isinstance(choice.topologies, Wildcard) for choice in choices):
        levels = tgt.get_facts(target.MemoryExecutionFacts).topologies
        expanded = []
        for choice in choices:
            if isinstance(choice.topologies, Wildcard):
                expanded.extend(replace(choice, topologies=(level,)) for level in levels)
            else:
                expanded.append(choice)
        choices = tuple(expanded)
        return choices[0] if len(choices) == 1 else Or(*choices)
    return declared


def sharded_on(execution_mesh: Mesh) -> Or:
    """A lexical tile, or a shard explicitly held by *execution_mesh*."""
    return Or(
        ShardLayout(
            layout=Wildcard(),
            attrs=Wildcard(),
            mesh=execution_mesh,
        ),
        Layout(),
    )


def selected(pattern, bindings: dict):
    """Resolve declaration branches fixed by already-bound attributes."""
    if isinstance(pattern, Tensor):
        return replace(pattern, layout=selected(pattern.layout, bindings))
    if isinstance(pattern, Switch) and pattern.param in bindings:
        branch = dict(pattern.branches).get(bindings[pattern.param])
        return None if branch is None else selected(branch, bindings)
    if isinstance(pattern, (And, Or)):
        parts = pattern.parts if isinstance(pattern, And) else pattern.patterns
        kept = tuple(held for part in parts if (held := selected(part, bindings)) is not None)
        if len(kept) == 1:
            return kept[0]
    return pattern


def variants(
    parameters: tuple[expr.ParamDef, ...],
    *,
    vary_defaulted: bool,
    values: Callable[[expr.ParamDef], tuple] | None = None,
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
        options = finite_options(param, values)
        if options is None:
            continue
        held = []
        for state in states:
            for value in options:
                matcher = Matcher(state)
                if matcher.match(param.pattern, value) and matcher.solve() is True:
                    held.append({**matcher.bindings, param.name: value})
        states = tuple(held)
    return states


def finite_options(param: expr.ParamDef, values) -> tuple | None:
    annotation = param.annotation
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return tuple(annotation)
    if annotation is expr.DType:
        return tuple(expr.dtype_members().values())
    return None if values is None else values(param)


def fixed_value(value, bindings: dict):
    """Resolve one declaration value when every symbolic leaf is bound."""
    if isinstance(value, Switch):
        if value.param not in bindings:
            return None
        return fixed_value(dict(value.branches).get(bindings[value.param]), bindings)
    if isinstance(value, tuple):
        parts = tuple(fixed_value(item, bindings) for item in value)
        return None if any(item is None for item in parts) else parts
    if value is None or type(value) in (int, str):
        return value
    name = getattr(value, "name", None)
    if name in bindings:
        return bindings[name]
    resolved = evaluated(value, bindings)
    return resolved if resolved is not None else value if name is None else None


def declared_shape(pattern: Tensor | None, bindings: dict) -> tuple | None:
    """Return a tensor declaration's concrete shape, when it states one."""
    if pattern is None or pattern.shape is None:
        return None
    shape = fixed_value(pattern.shape, bindings)
    return shape if isinstance(shape, tuple) and all(type(dim) is int for dim in shape) else None


def declared_symbol_axes(
    params: tuple, op, bindings: dict, name: str
) -> tuple[tuple[int, int], ...]:
    """Operand axes whose declared extent is the symbol named *name*."""
    axes = []
    for index, param in enumerate(params):
        pattern = param.pattern.read_on(op) if hasattr(param.pattern, "read_on") else param.pattern
        pattern = selected(pattern, bindings)
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


def mangle_variant_name(name: str, specializations: tuple[Pattern, ...]) -> str:
    if len(specializations) != 1 or not isinstance(specializations[0], Range):
        raise TypeError("variant requires exactly one pattern.Range")
    pattern = specializations[0]
    if not pattern.dim_var or pattern.lo is None or pattern.hi is None:
        raise TypeError("variant pattern.Range requires dim_var, lo, and hi")
    return f"{name}${pattern.dim_var}${pattern.lo}_{pattern.hi}"


__all__ = [
    "MOVED_STORAGES",
    "declared_execution_mesh",
    "declared_shape",
    "declared_symbol_axes",
    "dtype_place",
    "fixed_value",
    "locate_dim_var",
    "mangle_variant_name",
    "memory_execution_mesh",
    "operand_tile",
    "selected",
    "storage_place",
    "tensor_in",
    "thread_execution_mesh",
    "variants",
    "whole_vectors",
]
