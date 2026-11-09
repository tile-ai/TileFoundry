"""HIR ``verify_function`` enforcement of ``DimVar`` envelope rules.

A specialization is a claim about a range of runtime shapes, and a lowered
dispatch reads that range off a parameter at runtime. Both claims are only
checkable here: a forged envelope, a reference to a DimVar the signature does not
have, and a variant set that does not tile the envelope all lower to plausible
code and then select the wrong arm — or no arm — at runtime.
"""

from __future__ import annotations

import pytest

from tilefoundry import verifier
from tilefoundry.ir import core, expr, pattern, types


def _tensor(shape) -> types.TensorType:
    return types.TensorType(shape=shape, dtype=expr.DType.f32, layout=types.Layout(tuple(shape)), storage="gmem")


def _identity_fn(
    *,
    params: tuple[expr.Var, ...],
    return_type: types.TensorType | types.TupleType | None = None,
    specializations: tuple = (),
) -> core.Function:
    return core.Function.from_parts(
        name="f",
        params=params,
        body=params[0],
        return_type=return_type if return_type is not None else params[0].type,
        specializations=specializations,
    )


def test_a_signature_may_not_forge_its_dim_var_envelope() -> None:
    """Test a signature may not forge its dim var envelope.

    A specialization range stays inside the envelope of a parameter DimVar, and
    one name cannot carry two envelopes anywhere in a signature.
    Dispatch subjects lower to ``ShapeOf(param, axis)``, so a DimVar only
    in the return type is unreadable at runtime and remains unknown. The scan
    covers params, return type, and nested ``TupleType`` fields.
    """
    s = expr.DimVar(name="S_env", lo=1, hi=7)
    forged = _identity_fn(
        params=(expr.Var(type=_tensor((s,)), name="x"),),
        specializations=(pattern.Range("S_env", 0, 99),),
    )
    with pytest.raises(expr.VerifyError, match="not contained in DimVar envelope"):
        verifier.verify_function(forged)

    known = expr.Var(type=_tensor((expr.DimVar(name="S_known", lo=1, hi=7),)), name="x")
    with pytest.raises(expr.VerifyError, match="references unknown DimVar"):
        verifier.verify_function(
            _identity_fn(params=(known,), specializations=(pattern.Range("OTHER", 1, 3),))
        )

    r = expr.DimVar(name="R_ret_only", lo=1, hi=7)
    with pytest.raises(expr.VerifyError, match="references unknown DimVar"):
        verifier.verify_function(
            _identity_fn(
                params=(expr.Var(type=_tensor((4,)), name="x"),),
                return_type=_tensor((r,)),
                specializations=(pattern.Range("R_ret_only", 1, 3),),
            )
        )

    lo_var = expr.DimVar(name="S_inc", lo=1, hi=7)
    hi_var = expr.DimVar(name="S_inc", lo=4, hi=15)
    x = expr.Var(type=_tensor((lo_var,)), name="x")
    inconsistent = (
        _identity_fn(params=(x, expr.Var(type=_tensor((hi_var,)), name="y"))),
        _identity_fn(params=(x,), return_type=_tensor((hi_var,))),
        _identity_fn(params=(x,), return_type=types.TupleType(fields=(_tensor((hi_var,)),))),
    )
    for fn in inconsistent:
        with pytest.raises(expr.VerifyError, match="inconsistent DimVar bounds for 'S_inc'"):
            verifier.verify_function(fn)


def _dispatch_proto(name: str, env, ranges):
    """A dispatch prototype over ``DimVar`` with one variant per ```` in *ranges*.

    A dispatch prototype over ``DimVar(name, *env)`` with one variant per
    ``(lo, hi)`` in *ranges*. Used to exercise the closed-interval partition
    verifier (`verify_partition`).
    """
    s = expr.DimVar(name=name, lo=env[0], hi=env[1])
    ty = _tensor((s,))
    base = core.Function.from_parts(name="g", params=(expr.Var(type=ty, name="x"),), body=None, return_type=ty)
    for lo, hi in ranges:
        x = expr.Var(type=ty, name="x")
        base.add_variant(
            core.Function.from_parts(
                name="g",
                params=(x,),
                body=x,
                return_type=ty,
                specializations=(pattern.Range(name, lo, hi),),
            )
        )
    return base


def test_variants_must_tile_the_envelope_exactly() -> None:
    """The variants of a prototype partition its envelope: complete and disjoint.

    The variants of a prototype partition its envelope with closed ranges: complete and
    disjoint. A gap leaves a runtime shape with no arm, and an overlap makes the
    selected arm depend on evaluation order.
    """
    verifier.verify_function(_dispatch_proto("S_par_ok", (1, 7), [(1, 4), (5, 7)]))
    verifier.verify_function(_dispatch_proto("S_par_pt", (1, 7), [(1, 3), (4, 4), (5, 7)]))

    with pytest.raises(expr.VerifyError, match="gap or overlap at 4"):
        verifier.verify_function(_dispatch_proto("S_par_ov", (1, 7), [(1, 5), (4, 7)]))

    with pytest.raises(expr.VerifyError, match="gap or overlap at 5"):
        verifier.verify_function(_dispatch_proto("S_par_gap", (1, 7), [(1, 3), (5, 7)]))

    with pytest.raises(expr.VerifyError, match="gap or overlap at 6"):
        verifier.verify_function(_dispatch_proto("S_par_inc", (1, 7), [(1, 4), (6, 7)]))
