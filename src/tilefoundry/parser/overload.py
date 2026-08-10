"""Pattern-based overload resolution for OpSchema candidates.

Given a list of :class:`~tilefoundry.ir.core.op_schema.OpSchema` candidates
sharing the same ``(dialect, name)`` and the runtime arg types of a
call site, this module:

1. Filters out candidates whose arity is incompatible with the call.
2. For each remaining candidate, walks ``signature`` in order and runs
   each ParamDef's ``pattern`` against the corresponding arg type.
3. Returns the **first** candidate whose every input ParamDef pattern
   matches (F3 first-match lock).

It is intentionally minimal — it ships only what dispatch / parser
need to dispatch the new OpSchema-based Ops.
typeinfer / sugar / call-arg lifting stay in their original modules.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

from tilefoundry.ir.core.op_schema import OpSchema
from tilefoundry.ir.core.param_def import ParamDef

# --- Helpers ------------------------------------------------------------


def _input_params(schema: OpSchema) -> list[ParamDef]:
    """Return signature ParamDefs whose ``kind == 'input'`` in order."""
    return [pd for pd in schema.signature if pd.kind == "input"]


def _arity_window(schema: OpSchema) -> tuple[int, int]:
    """Return ``(min_required, max_accepted)`` arity window for inputs.

    Only inputs are considered for arg matching; attributes are bound by
    keyword and have their own validation path (out of scope here).

    Per spec, a param is omittable iff it has a default
    (``default is not MISSING``). ``optional`` is purely a *nullable*
    flag (whether the value type may be ``None``) and does **not**
    affect arity — ``optional=True, default=MISSING`` still requires
    the caller to pass an arg (the value just may be ``None``).
    """
    inputs = _input_params(schema)
    n_min = sum(1 for pd in inputs if pd.is_required)
    n_max = len(inputs)
    return n_min, n_max


def _pattern_matches(pd: ParamDef, arg_type: Any) -> bool:
    """True iff ``pd.pattern`` accepts ``arg_type`` (or no pattern given)."""
    if pd.pattern is None:
        return True
    return pd.pattern.match(arg_type)


# --- Public API ---------------------------------------------------------


class OverloadError(LookupError):
    """No OpSchema candidate matched the given arg types."""


def filter_candidates(
    candidates: Iterable[OpSchema], arg_types: Sequence[Any]
) -> list[OpSchema]:
    """Return candidates whose arity + every input pattern matches.

    Order is preserved; this is the raw filter without first-match
    selection. Useful for diagnostics / multi-match reporting.
    """
    n = len(arg_types)
    out: list[OpSchema] = []
    for schema in candidates:
        n_min, n_max = _arity_window(schema)
        if not (n_min <= n <= n_max):
            continue
        inputs = _input_params(schema)
        ok = True
        for pd, arg_ty in zip(inputs, arg_types):
            if not _pattern_matches(pd, arg_ty):
                ok = False
                break
        if ok:
            out.append(schema)
    return out


def resolve(
    candidates: Iterable[OpSchema], arg_types: Sequence[Any]
) -> OpSchema:
    """Return the first matching candidate (F3 first-match lock).

    Raises :class:`OverloadError` if no candidate matches.
    """
    matches = filter_candidates(candidates, arg_types)
    if not matches:
        cands = list(candidates)
        names = ", ".join(f"{s.dialect}:{s.name}" for s in cands) or "<none>"
        raise OverloadError(
            f"No OpSchema candidate matched arg types "
            f"{[type(t).__name__ for t in arg_types]!r}; tried: {names}"
        )
    return matches[0]


__all__ = [
    "OverloadError",
    "filter_candidates",
    "resolve",
]
