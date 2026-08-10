"""HIR ``verify_function`` enforcement of ``DimVar`` envelope rules.

A specialization is a claim about a range of runtime shapes, and a lowered
dispatch reads that range off a parameter at runtime. Both claims are only
checkable here: a forged envelope, a reference to a DimVar the signature does not
have, and a variant set that does not tile the envelope all lower to plausible
code and then select the wrong arm — or no arm — at runtime.
"""

from __future__ import annotations

import pytest

from tilefoundry.ir.core import Var, VerifyError
from tilefoundry.ir.core.pattern import DimVarRangePat
from tilefoundry.ir.hir.function import Function as HirFunction
from tilefoundry.ir.hir.verify import verify_function
from tilefoundry.ir.types import DType, TensorType
from tilefoundry.ir.types.dim import DimVar
from tilefoundry.ir.types.tensor_type import TupleType


def _tensor(shape) -> TensorType:
    return TensorType(shape=shape, dtype=DType.f32, layout=None, storage="gmem")


def _identity_fn(
    *,
    params: tuple[Var, ...],
    return_type: TensorType | TupleType | None = None,
    specializations: tuple = (),
) -> HirFunction:
    return HirFunction.build(
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
    ``DispatchCall.subject`` lowers to ``ShapeOf(param, axis)``, so a DimVar only
    in the return type is unreadable at runtime and remains unknown. The scan
    covers params, return type, and nested ``TupleType`` fields.
    """
    s = DimVar(name="S_env", lo=1, hi=8)
    forged = _identity_fn(
        params=(Var(type=_tensor((s,)), name="x"),),
        specializations=(DimVarRangePat("S_env", 0, 100),),
    )
    with pytest.raises(VerifyError, match="not contained in DimVar envelope"):
        verify_function(forged)

    known = Var(type=_tensor((DimVar(name="S_known", lo=1, hi=8),)), name="x")
    with pytest.raises(VerifyError, match="references unknown DimVar"):
        verify_function(
            _identity_fn(params=(known,), specializations=(DimVarRangePat("OTHER", 1, 4),))
        )

    r = DimVar(name="R_ret_only", lo=1, hi=8)
    with pytest.raises(VerifyError, match="references unknown DimVar"):
        verify_function(
            _identity_fn(
                params=(Var(type=_tensor((4,)), name="x"),),
                return_type=_tensor((r,)),
                specializations=(DimVarRangePat("R_ret_only", 1, 4),),
            )
        )

    lo_var = DimVar(name="S_inc", lo=1, hi=8)
    hi_var = DimVar(name="S_inc", lo=4, hi=16)
    x = Var(type=_tensor((lo_var,)), name="x")
    inconsistent = (
        _identity_fn(params=(x, Var(type=_tensor((hi_var,)), name="y"))),
        _identity_fn(params=(x,), return_type=_tensor((hi_var,))),
        _identity_fn(params=(x,), return_type=TupleType(fields=(_tensor((hi_var,)),))),
    )
    for fn in inconsistent:
        with pytest.raises(VerifyError, match="inconsistent DimVar bounds for 'S_inc'"):
            verify_function(fn)


def _dispatch_proto(name: str, env, ranges):
    """A dispatch prototype over ``DimVar`` with one variant per ```` in *ranges*.

    A dispatch prototype over ``DimVar(name, *env)`` with one variant per
    ``(lo, hi)`` in *ranges*. Used to exercise the half-open-interval partition
    verifier (`_verify_partition`).
    """
    s = DimVar(name=name, lo=env[0], hi=env[1])
    ty = _tensor((s,))
    base = HirFunction.build(name="g", params=(Var(type=ty, name="x"),), body=None, return_type=ty)
    for lo, hi in ranges:
        x = Var(type=ty, name="x")
        base.add_variant(
            HirFunction.build(
                name="g",
                params=(x,),
                body=x,
                return_type=ty,
                specializations=(DimVarRangePat(name, lo, hi),),
            )
        )
    return base


def test_variants_must_tile_the_envelope_exactly() -> None:
    """The variants of a prototype partition its half-open envelope: complete and disjoint.

    The variants of a prototype partition its half-open envelope: complete and
    disjoint. A gap leaves a runtime shape with no arm, and an overlap makes the
    selected arm depend on evaluation order.
    """
    verify_function(_dispatch_proto("S_par_ok", (1, 8), [(1, 5), (5, 8)]))
    verify_function(_dispatch_proto("S_par_pt", (1, 8), [(1, 4), (4, 5), (5, 8)]))

    with pytest.raises(VerifyError, match="gap or overlap at 4"):
        verify_function(_dispatch_proto("S_par_ov", (1, 8), [(1, 5), (4, 8)]))

    with pytest.raises(VerifyError, match="gap or overlap at 5"):
        verify_function(_dispatch_proto("S_par_gap", (1, 8), [(1, 3), (5, 8)]))

    with pytest.raises(VerifyError, match=r"cover \[1, 7\) but the envelope is \[1, 8\)"):
        verify_function(_dispatch_proto("S_par_inc", (1, 8), [(1, 5), (5, 7)]))
