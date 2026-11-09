"""Execution regions turn one unit's work into whole-scope cost."""

from __future__ import annotations

import pytest

from tilefoundry.dsl import *
from tilefoundry.interfaces import typeinfer
from tilefoundry.ir import core, expr


@pytest.mark.parametrize("kind", (core.MeshRegion, core.LoopRegion))
def test_analysis_rejects_a_region_body_that_embeds_its_raw_argument(kind) -> None:
    """The isolation invariant is checked on hand-built HIR as an analysis consumer."""
    value_type = TensorType(shape=(8,), dtype=DType.f32, layout=None, storage=StorageKind.GMEM)
    value = expr.Var(name="value", type=value_type)
    fields = (
        {"mesh": Mesh((Topology("cta", 1),), Layout((1,), (1,)), ("cta",))}
        if kind is core.MeshRegion
        else {
            "induction_var": expr.Var(name="i", type=TensorType.from_scalar(DType.i64)),
            "yield_values": (),
            "extent": 2,
            "step": 1,
        }
    )
    scope = kind(
        **fields,
        params=(expr.Var(name="param", type=value_type),),
        args=(value,),
        body=value,
        type=value_type,
    )

    with pytest.raises(expr.VerifyError, match="region is not isolated"):
        typeinfer.TypeInferVisitor().visit(scope, typeinfer.TypeInferContext())
