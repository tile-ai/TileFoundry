from __future__ import annotations

from tilefoundry.ir.core import Expr, TypeInferContext, VerifyError
from tilefoundry.ir.core.expr import Call, Var
from tilefoundry.ir.core.pattern import DimVarRangePat
from tilefoundry.ir.tir.stmt import Stmt
from tilefoundry.ir.types import TensorType
from tilefoundry.ir.types.dim import DimVar
from tilefoundry.ir.types.tensor_type import TupleType

from .function import Function, canonical_specialization_signature


def verify_function(fn: Function) -> None:
    """Verify one hir ``Function``: params, signature ``DimVar``s, and its shape-specific rules (normal body / dispatch prototype / variant)."""
    for p in fn.params:
        if not isinstance(p, Var):
            raise VerifyError(f"hir Function {fn.name!r}: params must be Vars")
    _verify_signature_dim_vars(fn)
    if fn.variants:
        if fn.body is not None:
            raise VerifyError(
                f"hir Function {fn.name!r}: a function with variants must have "
                f"no body (a dispatch prototype's body is None / `pass`)"
            )
        _verify_variants(fn)
        return
    if fn.body is None:
        # Unsealed authoring transient: a `pass` prototype before its first
        # `.specialize`. verify_module rejects this at sealed Module scope.
        return
    if not isinstance(fn.body, Expr):
        raise VerifyError(
            f"hir Function {fn.name!r}: body must be an Expr or None, got "
            f"{type(fn.body).__name__}"
        )
    _reject_stmt_nodes(fn.body)
    # Drive typeinfer on the whole body (cache populates as side effect).
    TypeInferContext().type_of(fn.body)


def _verify_variants(base: Function) -> None:
    """Verify a dispatch prototype's variants and their envelope partition."""
    base_param_types = tuple(p.type for p in base.params)
    sigs: dict[str, Function] = {}
    for v in base.variants:
        if v.variants:
            raise VerifyError(
                f"hir Function {v.name!r}: a variant must not carry variants "
                f"(specialization nesting is one level)"
            )
        if not v.specializations:
            raise VerifyError(
                f"hir Function {base.name!r}: a variant must carry a "
                f"specialization pattern"
            )
        if v.body is None:
            raise VerifyError(
                f"hir Function {base.name!r}: a variant must have a real body, "
                f"not `pass`"
            )
        if v.name != base.name:
            raise VerifyError(
                f"variant name {v.name!r} does not match base {base.name!r}"
            )
        if tuple(p.type for p in v.params) != base_param_types or (
            v.return_type != base.return_type
        ):
            raise VerifyError(
                f"hir Function {base.name!r}: variant signature must match the "
                f"base signature"
            )
        if v.target != base.target or tuple(v.topologies) != tuple(base.topologies):
            raise VerifyError(
                f"hir Function {base.name!r}: variant target / topologies must "
                f"match the base"
            )
        sig = canonical_specialization_signature(v.specializations)
        if sig in sigs:
            raise VerifyError(
                f"hir Function {base.name!r}: duplicate variant canonical "
                f"signature {sig!r}"
            )
        sigs[sig] = v
        verify_function(v)
    _verify_partition(base)


def _verify_partition(base: Function) -> None:
    """Check the variants' ranges partition the base DimVar envelope —
    pairwise disjoint and jointly complete over the half-open ``[lo, hi)``."""
    dim_vars: set[str] = set()
    ranges: list[tuple[int, int]] = []
    for v in base.variants:
        for pat in v.specializations:
            if not isinstance(pat, DimVarRangePat):
                raise VerifyError(
                    f"hir Function {base.name!r}: only DimVarRangePat is "
                    f"supported for dispatch (got {type(pat).__name__})"
                )
            dim_vars.add(pat.dim_var)
            ranges.append((pat.lo, pat.hi))
    if len(dim_vars) != 1:
        raise VerifyError(
            f"hir Function {base.name!r}: variants must dispatch on a single "
            f"DimVar, got {sorted(dim_vars)}"
        )
    envelope = _collect_param_dim_vars(base).get(next(iter(dim_vars)))
    if envelope is None:
        raise VerifyError(
            f"hir Function {base.name!r}: dispatch DimVar "
            f"{next(iter(dim_vars))!r} is not reachable from an input parameter"
        )
    lo, hi = envelope
    # Half-open intervals: an adjacent pair is ``[.., c)`` then ``[c, ..)`` —
    # the next range starts at the previous range's exclusive end.
    cursor = lo
    for rlo, rhi in sorted(ranges):
        if rlo != cursor:
            raise VerifyError(
                f"hir Function {base.name!r}: variant ranges do not partition "
                f"envelope [{lo}, {hi}) — gap or overlap at {rlo} (expected "
                f"{cursor})"
            )
        cursor = rhi
    if cursor != hi:
        raise VerifyError(
            f"hir Function {base.name!r}: variant ranges cover "
            f"[{lo}, {cursor}) but the envelope is [{lo}, {hi})"
        )


def _ingest_dim_vars(ty: object, bounds: dict[str, tuple[int, int]]) -> None:
    """Recurse into ``TensorType`` / ``TupleType`` collecting ``DimVar``
    bounds. Raises ``VerifyError`` if two same-name ``DimVar``s disagree
    on ``(lo, hi)``.

    Shape entries may be direct ``DimVar`` instances, ``int`` literals,
    ``Constant`` scalars, or dim-arithmetic ``Call`` trees built from
    ``DimAdd`` / ``DimSub`` / ``DimMul`` / ``DimFloorDiv`` / ``DimMod`` /
    ``DimMin`` / ``DimMax`` (see ``tilefoundry.ir.types.shape_dim``). The
    walker recurses into ``Call.args`` to collect every ``DimVar``
    reachable through dim expressions, so a shape like
    ``[1, 2, DimAdd(CTX_LEN, 1), 256]`` still anchors the signature's
    ``CTX_LEN`` bound for envelope / consistency checks.
    """
    if isinstance(ty, TensorType):
        for entry in ty.shape:
            _ingest_shape_entry(entry, bounds)
    elif isinstance(ty, TupleType):
        for field in ty.fields:
            _ingest_dim_vars(field, bounds)


def _ingest_shape_entry(entry: object, bounds: dict[str, tuple[int, int]]) -> None:
    if isinstance(entry, DimVar):
        prior = bounds.get(entry.name)
        if prior is None:
            bounds[entry.name] = (entry.lo, entry.hi)
        elif prior != (entry.lo, entry.hi):
            raise VerifyError(
                f"inconsistent DimVar bounds for {entry.name!r} within "
                f"function signature: [{prior[0]}, {prior[1]}) vs "
                f"[{entry.lo}, {entry.hi})"
            )
        return
    if isinstance(entry, Call):
        for arg in entry.args:
            _ingest_shape_entry(arg, bounds)


def _collect_param_dim_vars(fn: Function) -> dict[str, tuple[int, int]]:
    """Bounds from ``fn.params`` only.

    Drives envelope ⊆ and unknown-name checks: specializations must
    anchor to a ``DimVar`` reachable from an *input* param, because
    ``DispatchCall.subject`` lowers to ``ShapeOf(param, axis)`` and
    can only reference a value the caller provides.
    """
    bounds: dict[str, tuple[int, int]] = {}
    for p in fn.params:
        _ingest_dim_vars(p.type, bounds)
    return bounds


def _check_signature_dim_var_consistency(fn: Function) -> None:
    """Scan params + return_type for same-name ``DimVar`` consistency.

    Recurses through ``TensorType`` and ``TupleType``. Raises on any
    same-name disagreement anywhere in the signature.
    """
    bounds: dict[str, tuple[int, int]] = {}
    for p in fn.params:
        _ingest_dim_vars(p.type, bounds)
    _ingest_dim_vars(getattr(fn, "return_type", None), bounds)


def _verify_signature_dim_vars(fn: Function) -> None:
    # Order: consistency first (raises on any same-name disagreement
    # across params + return), then envelope-against-params (envelope
    # anchor restricted to input params).
    _check_signature_dim_var_consistency(fn)
    param_bounds = _collect_param_dim_vars(fn)
    for pat in fn.specializations:
        if not isinstance(pat, DimVarRangePat):
            continue
        dv_bounds = param_bounds.get(pat.dim_var)
        if dv_bounds is None:
            raise VerifyError(
                f"specialization DimVarRangePat({pat.dim_var!r}, {pat.lo}, "
                f"{pat.hi}) references unknown DimVar (specializations must "
                f"anchor to a DimVar reachable from an input parameter)"
            )
        lo, hi = dv_bounds
        if not (lo <= pat.lo and pat.hi <= hi):
            raise VerifyError(
                f"DimVarRangePat ({pat.dim_var!r}, {pat.lo}, {pat.hi}) is not "
                f"contained in DimVar envelope [{lo}, {hi})"
            )


def _reject_stmt_nodes(expr: Expr) -> None:
    if isinstance(expr, Stmt):
        raise VerifyError(f"hir body contains a Stmt node {type(expr).__name__}; hir is expr-only")
    if isinstance(expr, Call):
        for arg in expr.args:
            _reject_stmt_nodes(arg)
    # Var / Constant have no sub-Expr children.


__all__ = ["verify_function"]
