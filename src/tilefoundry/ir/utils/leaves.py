"""Where the tensor leaves of a type sit, flat, in the order a reader indexes them."""

from __future__ import annotations

import operator
from dataclasses import replace

from tilefoundry.ir import types

__all__ = ["leaf_rank", "leaf_span", "leaves_of", "map_tensor_types", "result_fields", "tensor_types"]


def leaves_of(type_: "types.Type") -> tuple:
    """Every tensor leaf of one value, flat, in the order a reader indexes them."""
    if isinstance(type_, types.TupleType):
        return tuple(leaf for field in type_.fields for leaf in leaves_of(field))
    return (type_,) if isinstance(type_, types.TensorType) else ()


def result_fields(type_: "types.Type") -> tuple:
    """The result boundaries of one value: a tuple's fields, a nested tuple's own leaves in place.

    Boundaries count as ``core.SSAValue`` indexes do, so a field holding a tuple
    is as many boundaries as it has leaves. ``TupleType.Void`` is one boundary:
    an effect Op states one relation for it.
    """
    if isinstance(type_, types.TupleType) and type_ != types.TupleType.Void:
        return tuple(
            boundary
            for field in type_.fields
            for boundary in (leaves_of(field) if isinstance(field, types.TupleType) else (field,))
        )
    return (type_,)


def leaf_span(type_: "types.Type", field: int) -> "tuple[int, int]":
    """Where one field of a value begins among its flat leaves, and how many.

    A structured value is indexed by leaf, not by top-level field, so a field
    holding a tuple of its own covers a run of them. Whoever takes that field
    takes that run.
    """
    if not isinstance(type_, types.TupleType):
        return (0, len(leaves_of(type_)))
    begin = sum(len(leaves_of(held)) for held in type_.fields[:field])
    return (begin, len(leaves_of(type_.fields[field])))


def leaf_rank(type_) -> int:
    """Number of coordinates in a scalar or tensor leaf."""
    return len(type_.shape) if isinstance(type_, types.TensorType) else 1


def tensor_types(type: types.Type) -> tuple[types.TensorType, ...]:
    """The tensor leaves of *type*, flattened out of tuple nesting."""
    if isinstance(type, types.TensorType):
        return (type,)
    if isinstance(type, types.TupleType):
        return tuple(leaf for field in type.fields for leaf in tensor_types(field))
    return ()


def map_tensor_types(type_: types.Type, fn) -> types.Type:
    """*type_* with *fn* applied to each tensor leaf, tuple nesting kept."""
    if isinstance(type_, types.TensorType):
        return fn(type_)
    if isinstance(type_, types.TupleType):
        fields = tuple(map_tensor_types(field, fn) for field in type_.fields)
        return type_ if all(map(operator.is_, fields, type_.fields)) else replace(type_, fields=fields)
    return type_
