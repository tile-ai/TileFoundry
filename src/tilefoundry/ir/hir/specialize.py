"""Choosing which implementation runs, and at what size.

A function authored for decode says two things at once: that its context length
is any length in a range, and that different parts of that range want different
implementations. Running it needs neither -- it needs one implementation at one
length. Both choices are made here.

They are separate steps because they fail differently. Picking the wrong
implementation is a dispatch error; picking a length the source never admitted
is a shape error. Collapsing them would report one as the other.
"""

from __future__ import annotations

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


#: Where a derived function came from, recorded on the function itself.
#:
#: A rebuilt function is a new object, so the only thing about it that looks
#: familiar is its name -- and a name is shared by anything anybody chose to
#: call the same. Ownership answered by name would therefore accept a function
#: from another module entirely, which is the opposite of what asking about
#: ownership is for. The origin is stamped here instead, by the one function
#: that produces a derived instance, so a caller cannot arrive holding it by
#: accident.
PROVENANCE = "_specialized_from"

#: The extents a derived function was built at, recorded beside its origin.
#:
#: The origin alone does not say which size was chosen, and neither does the
#: resulting signature: a dimension can occur only in a loop bound or a body
#: operation's attribute, so two different sizes of one function can be rebuilt
#: into two different programs whose parameters and return type are identical.
#: Anything asking whether two derived functions are the same program has to
#: compare the bindings themselves, so they are written down here rather than
#: inferred from what the rebuild happened to change.
BOUND_DIMS = "_specialized_dims"


def _record_provenance(
    derived: Function, origin: Function, dims: Mapping[str, int]
) -> None:
    """Note that *derived* is *origin* at *dims*.

    Written through `object.__setattr__` because a Function is frozen; this is
    the same authoring-phase mutation `seal` and `add_variant` use, and it does
    not participate in equality, so two functions specialised from different
    origins are still equal when they are the same program.

    The extents are stored sorted by name, so one set of bindings has one
    representation and two records can be compared directly rather than each
    caller having to canonicalise first.
    """
    object.__setattr__(derived, PROVENANCE, origin)
    object.__setattr__(derived, BOUND_DIMS, tuple(sorted(dims.items())))


def origin_of(function: object) -> Function | None:
    """The function *function* was specialised from, if it was."""
    return getattr(function, PROVENANCE, None)


def bound_dims_of(function: object) -> tuple[tuple[str, int], ...] | None:
    """The extents *function* was specialised at, if it was, sorted by name."""
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

    matching = [
        variant
        for variant in fn.variants
        if _covers(fn, variant, dims)
    ]
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
        raise SpecializationError(
            f"specialising {fn.name!r} needs at least one dimension to bind"
        )
    chosen = variant_for(fn, dims)
    if chosen.body is None:
        raise SpecializationError(
            f"{fn.name!r} resolved to a variant with no body; a dispatch "
            "prototype states which implementation to use, not what it does"
        )

    # Everything the implementation reaches, not just what its signature
    # names. A dimension can be introduced inside the body -- a reshape that
    # states a block length -- and refusing to bind it because the parameters
    # never mention it would make exactly those functions unspecialisable.
    present = set(residual_dims(chosen))
    for pattern in chosen.specializations:
        if isinstance(pattern, DimVarRangePat):
            present.add(pattern.dim_var)
    unknown = sorted(set(dims) - present)
    if unknown:
        raise SpecializationError(
            f"{fn.name!r} has no dimension named {unknown!r}; it states "
            f"{sorted(present)}"
        )

    # Whether to rebuild is decided by where the dimension occurs, not by
    # whether the signature moved. A loop extent, an operation's shape
    # attribute, a return type, a callee -- a dimension in any of those is one
    # to substitute, and a function whose parameters happen not to mention it
    # would otherwise be handed back still holding a range.
    if not set(dims) & set(residual_dims(chosen)):
        return chosen
    bound = tuple(substitute_dims(param.type, dims) for param in chosen.params)
    derived = _elaborate_from_bound_types(
        chosen, bound, ctx if ctx is not None else TypeInferContext(), dims=dims
    )
    _record_provenance(derived, chosen, dims)
    return derived


def specialize_concretely(fn: Function, dims: Mapping[str, int]) -> Function:
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
            raise SpecializationError(
                f"specialising {fn.name!r}: {name!r} is not a dimension name"
            )
        if isinstance(extent, bool) or not isinstance(extent, int):
            raise SpecializationError(
                f"specialising {fn.name!r}: dimension {name!r} takes an integer "
                f"extent, got {extent!r}"
            )
    concrete = specialize_function(fn, dims)
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
    return not residual_dims(fn) and not has_symbolic_dims(fn.return_type)


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
