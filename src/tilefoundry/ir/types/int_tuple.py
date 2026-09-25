"""IntTuple alias + helpers."""

from __future__ import annotations

from typing import Union, overload

from tilefoundry.ir.types.tensor_type import ShapeDim

IntTuple = Union[int, tuple["IntTuple", ...]]


@overload
def flatten(t: IntTuple) -> tuple[int, ...]: ...


@overload
def flatten(t: object) -> tuple[object, ...]: ...


def flatten(t: object) -> tuple[object, ...]:
    if not isinstance(t, tuple):
        return (t,)
    return tuple(value for item in t for value in flatten(item))


def product(t) -> "ShapeDim":
    from .mesh import Topology  # noqa: PLC0415

    result = 1
    for v in flatten(t):
        if isinstance(v, Topology):
            v = v.size
        result *= v
    return result


def repeat_like(profile, value) -> object:
    """CuTe ``repeat_like``: *value* at every leaf, nested like *profile*.

    What a flat tuple has to say about a nested one is said by building it
    against the nesting and flattening that, rather than by working out which
    flat positions each mode covers.
    """
    if not isinstance(profile, tuple):
        return value
    return tuple(repeat_like(item, value) for item in profile)


def _unflatten(flat: tuple, profile) -> tuple:
    """Take *profile*'s worth of *flat*, returning it nested and what is left."""
    if not isinstance(profile, tuple):
        if not flat:
            raise ValueError("unflatten: the profile asks for more modes than the tuple has")
        return flat[0], flat[1:]
    nested: list = []
    for item in profile:
        value, flat = _unflatten(flat, item)
        nested.append(value)
    return tuple(nested), flat


def unflatten(flat: tuple, profile) -> tuple:
    """CuTe ``unflatten``: nest a flat tuple to *profile*'s structure.

    Only *profile*'s nesting is read, never its leaves, so the profile may be
    the grouping itself. ``flatten(unflatten(t, p)) == t``.
    """
    nested, rest = _unflatten(flat, profile)
    if rest:
        raise ValueError(
            f"unflatten: the profile accounts for {len(flat) - len(rest)} of the "
            f"tuple's {len(flat)} modes"
        )
    return nested


__all__ = ["IntTuple", "flatten", "product", "repeat_like", "unflatten"]
