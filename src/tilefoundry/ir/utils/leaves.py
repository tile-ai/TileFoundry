"""Where the tensor leaves of a type sit, flat, in the order a reader indexes them."""

from __future__ import annotations

from tilefoundry.ir import types

__all__ = ["leaf_rank", "leaf_span", "leaves_of", "tensor_types"]


def leaves_of(type_: "types.Type") -> tuple:
    """Every tensor leaf of one value, flat, in the order a reader indexes them."""
    if isinstance(type_, types.TupleType):
        return tuple(leaf for field in type_.fields for leaf in leaves_of(field))
    return (type_,) if isinstance(type_, types.TensorType) else ()


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
