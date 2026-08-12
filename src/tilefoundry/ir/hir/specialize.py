"""Choose a specialization variant and bind its symbolic extents.

Derived functions record their origin and sorted bindings because signatures
need not expose every bound dimension. Display labels remain outside equality,
hashing, and canonical printing. Dispatch and shape binding stay separate so
their failures remain distinguishable.

See [hir §2](docs/spec/hir.md#2-function-specialization-api).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping

from tilefoundry.ir.core.pattern import DimVarRangePat
from tilefoundry.ir.types.substitute import (
    dim_vars_by_name,
    has_symbolic_dims,
    substitute_dims,
)
from tilefoundry.visitor_registry.contexts import TypeInferContext

from .function import Function, _elaborate_from_bound_types


class SpecializationError(ValueError):
    """A function cannot be specialised as asked."""


PROVENANCE = "_specialized_from"

BOUND_DIMS = "_specialized_dims"

DISPLAY_NAME = "_display_name"


def display_name(fn: Function) -> str | None:
    """This variant's label, or ``None`` where its author gave none."""
    return getattr(fn, DISPLAY_NAME, None)


def _record_provenance(
    derived: Function, origin: Function, dims: Mapping[str, int] | None
) -> None:
    """Note that *derived* is *origin*, at *dims* when a size was chosen.

    Written through `object.__setattr__` because a Function is frozen; this is
    the same authoring-phase mutation `seal` and `add_variant` use, and it does
    not participate in equality, so two functions specialised from different
    origins are still equal when they are the same program.

    Extents are stored sorted by name, so one binding set has one representation.
    A rebuild that chose none records none rather than an empty set, which would
    compare equal to another such rebuild's.
    """
    object.__setattr__(derived, PROVENANCE, origin)
    if dims is not None:
        object.__setattr__(derived, BOUND_DIMS, tuple(sorted(dims.items())))


def _record_complete_bindings(
    function: Function, dims: Mapping[str, int]
) -> Function:
    """Record a public call's complete program bindings on a derived Function."""
    if bound_dims_of(function) is None:
        derived = dataclasses.replace(function)
        _record_provenance(derived, function, dims)
        return derived
    object.__setattr__(function, BOUND_DIMS, tuple(sorted(dims.items())))
    return function


def origin_of(function: object) -> Function | None:
    """The function *function* was rebuilt from, if it was."""
    return getattr(function, PROVENANCE, None)


def bound_dims_of(function: object) -> tuple[tuple[str, int], ...] | None:
    """The extents *function* was rebuilt at, if any were chosen, sorted by name."""
    return getattr(function, BOUND_DIMS, None)


def variant_for(fn: Function, dims: Mapping[str, int]) -> Function:
    """The one implementation of *fn* that covers *dims*.

    A function with no variants is its own implementation. Otherwise exactly
    one variant must claim the stated extents: none means the source never
    covered this size, and more than one means the source contradicts itself.
    Neither is something to resolve by picking an order.
    """
    if not fn.variants:
        return fn

    matching = [variant for variant in fn.variants if _covers(fn, variant, dims)]
    if len(matching) == 1:
        return matching[0]
    stated = ", ".join(f"{name}={value}" for name, value in sorted(dims.items()))
    if not matching:
        raise SpecializationError(
            f"{fn.name!r} declares no variant covering {stated or 'anything'}; "
            f"its variants cover {_coverage(fn)}"
        )
    raise SpecializationError(
        f"{fn.name!r} has {len(matching)} variants covering {stated}; "
        f"they cover {_coverage(fn)} and a size may belong to only one"
    )


def _covers(fn: Function, variant: Function, dims: Mapping[str, int]) -> bool:
    """Whether *variant*'s every stated range admits the chosen extents.

    A pattern the caller said nothing about is refused rather than skipped: a
    variant is selected by the dimensions it names, so an unstated one means
    the caller does not yet know which implementation they are asking for.
    """
    for pattern in variant.specializations:
        if not isinstance(pattern, DimVarRangePat):
            continue
        if pattern.dim_var not in dims:
            raise SpecializationError(
                f"{fn.name!r} selects a variant on {pattern.dim_var!r}, which "
                f"was not given a size; state it to choose an implementation"
            )
        if not pattern.lo <= dims[pattern.dim_var] < pattern.hi:
            return False
    return True


def _coverage(fn: Function) -> str:
    return "; ".join(
        ", ".join(
            f"{pattern.dim_var} in [{pattern.lo}, {pattern.hi})"
            for pattern in variant.specializations
            if isinstance(pattern, DimVarRangePat)
        )
        or "everything"
        for variant in fn.variants
    )


def specialize_function(
    fn: Function,
    dims: Mapping[str, int],
    *,
    ctx: TypeInferContext | None = None,
) -> Function:
    """*fn* at the stated extents: one implementation, its ranges resolved.

    The dimensions are checked against what the function actually has before
    anything is rebuilt. A name nothing uses is a caller who believes they
    specialised something, and quietly substituting nothing would let them
    carry on believing it.
    """
    if not dims:
        raise SpecializationError(f"specialising {fn.name!r} needs at least one dimension to bind")
    chosen = variant_for(fn, dims)
    if chosen.body is None:
        raise SpecializationError(
            f"{fn.name!r} resolved to a variant with no body; a dispatch "
            "prototype states which implementation to use, not what it does"
        )

    present = set(residual_dims(chosen))
    for pattern in chosen.specializations:
        if isinstance(pattern, DimVarRangePat):
            present.add(pattern.dim_var)
    unknown = sorted(set(dims) - present)
    if unknown:
        raise SpecializationError(
            f"{fn.name!r} has no dimension named {unknown!r}; it states {sorted(present)}"
        )

    if not set(dims) & set(residual_dims(chosen)):
        return chosen
    bound = tuple(substitute_dims(param.type, dims) for param in chosen.params)
    return _elaborate_from_bound_types(
        chosen, bound, ctx if ctx is not None else TypeInferContext(), dims=dims
    )


def specialize_concretely(
    fn: Function, dims: Mapping[str, int], ctx: TypeInferContext | None = None
) -> Function:
    """*fn* at the stated extents, with nothing left as a range.

    The stricter half of `specialize_function`, for callers that go on to run
    something over the result. Partial binding is useful when the choices are
    still being made one at a time; it is useless to anything that has to count
    elements, so a dimension left unbound is refused here rather than surfacing
    later as an extent that is not a number.
    """
    if not isinstance(dims, Mapping) or not dims:
        raise SpecializationError(
            f"specialising {fn.name!r} needs a non-empty mapping of dimension "
            f"names to extents, got {dims!r}"
        )
    for name, extent in dims.items():
        if not isinstance(name, str) or not name:
            raise SpecializationError(f"specialising {fn.name!r}: {name!r} is not a dimension name")
        if isinstance(extent, bool) or not isinstance(extent, int):
            raise SpecializationError(
                f"specialising {fn.name!r}: dimension {name!r} takes an integer "
                f"extent, got {extent!r}"
            )
    concrete = specialize_function(fn, dims, ctx=ctx)
    residual = residual_dims(concrete)
    if residual:
        raise SpecializationError(
            f"{fn.name!r} still states {list(residual)} as ranges after binding "
            f"{sorted(dims)}; every dimension has to be given an extent"
        )
    return concrete


def residual_dims(fn: Function) -> tuple[str, ...]:
    """Every dimension still stated as a range anywhere *fn* reaches.

    Signature, body, the shape-valued attributes its operations carry, the
    bounds of its loops, and the same again for every function it calls. A
    dimension can be introduced deep inside a callee -- a reshape that states a
    block length the caller never mentions -- and a scan that stopped at the
    signature would call such a function concrete while it still holds a range.
    """
    return tuple(dim_vars_reached(fn))


def dim_vars_reached(fn: Function) -> dict[str, object]:
    """The declarations behind `residual_dims`, by name.

    Same traversal, keeping the `DimVar` rather than only its name, for a caller
    that has to restate the bounds a dimension was declared with.
    """
    found: dict[str, object] = {}
    _walk_function(fn, found, set())
    return found


def _walk_function(fn: Function, found: dict[str, object], seen: set[int]) -> None:
    if id(fn) in seen:
        return
    seen.add(id(fn))
    for param in fn.params:
        found.update(dim_vars_by_name(param.type))
    found.update(dim_vars_by_name(fn.return_type))
    for variant in fn.variants:
        _walk_function(variant, found, seen)
    if fn.body is not None:
        _walk(fn.body, found, seen)


def _walk(expr: object, found: dict[str, object], seen: set[int], depth: int = 0) -> None:
    from tilefoundry.ir.types.substitute import _collect  # noqa: PLC0415

    if expr is None or depth > 256:
        return
    _collect(getattr(expr, "type", None), found)
    target = getattr(expr, "target", None)
    if isinstance(target, Function):
        _walk_function(target, found, seen)
    elif target is not None:
        for attribute in getattr(type(target), "params", lambda: ())():
            if attribute.kind == "attribute":
                _collect_entries(getattr(target, attribute.name, None), found)
    for name in ("args", "elements", "init_args", "yield_values", "carried_args"):
        for child in getattr(expr, name, ()) or ():
            _walk(child, found, seen, depth + 1)
    for bound in ("extent", "step", "start"):
        _collect_entries(getattr(expr, bound, None), found)
    _walk(getattr(expr, "body", None), found, seen, depth + 1)


def _collect_entries(value: object, found: dict[str, object]) -> None:
    from tilefoundry.ir.types.substitute import _collect  # noqa: PLC0415

    if isinstance(value, tuple):
        for entry in value:
            _collect(entry, found)
        return
    _collect(value, found)


def is_concrete(fn: Function) -> bool:
    """Whether *fn* states extents everywhere a size is required."""
    return not _function_has_symbolic_dims(fn, set())


def _function_has_symbolic_dims(fn: Function, seen: set[int]) -> bool:
    if id(fn) in seen:
        return False
    seen.add(id(fn))
    if any(has_symbolic_dims(param.type) for param in fn.params):
        return True
    if has_symbolic_dims(fn.return_type):
        return True
    if any(_function_has_symbolic_dims(variant, seen) for variant in fn.variants):
        return True
    return _expr_has_symbolic_dims(fn.body, seen)


def _expr_has_symbolic_dims(expr: object, seen: set[int], depth: int = 0) -> bool:
    if expr is None or depth > 256:
        return False
    if has_symbolic_dims(getattr(expr, "type", None)):
        return True
    target = getattr(expr, "target", None)
    if isinstance(target, Function):
        if _function_has_symbolic_dims(target, seen):
            return True
    elif target is not None:
        for attribute in getattr(type(target), "params", lambda: ())():
            if attribute.kind == "attribute" and has_symbolic_dims(
                getattr(target, attribute.name, None)
            ):
                return True
    for name in ("args", "elements", "init_args", "yield_values", "carried_args"):
        if any(
            _expr_has_symbolic_dims(child, seen, depth + 1)
            for child in getattr(expr, name, ()) or ()
        ):
            return True
    if any(
        has_symbolic_dims(getattr(expr, bound, None))
        for bound in ("extent", "step", "start")
    ):
        return True
    return _expr_has_symbolic_dims(getattr(expr, "body", None), seen, depth + 1)


__all__ = [
    "BOUND_DIMS",
    "PROVENANCE",
    "SpecializationError",
    "bound_dims_of",
    "dim_vars_reached",
    "is_concrete",
    "origin_of",
    "residual_dims",
    "specialize_concretely",
    "specialize_function",
    "variant_for",
]
