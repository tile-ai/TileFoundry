"""Target-neutral tests and helpers."""

from __future__ import annotations

import pytest

from tilefoundry.dsl import *
from tilefoundry.ir.core import Var, VerifyError
from tilefoundry.ir.hir.loop_region import LoopRegion
from tilefoundry.ir.hir.mesh_region import MeshRegion
from tilefoundry.visitor_registry.contexts import TypeInferContext
from tilefoundry.visitor_registry.typeinfer import TypeInferVisitor


@pytest.mark.parametrize("kind", (MeshRegion, LoopRegion))
def test_analysis_rejects_a_region_body_that_embeds_its_raw_argument(kind) -> None:
    """The isolation invariant is checked on hand-built HIR as an analysis consumer."""
    value_type = TensorType(shape=(8,), dtype=DType.f32, layout=None, storage=StorageKind.GMEM)
    value = Var(name="value", type=value_type)
    fields = (
        {"mesh": Mesh((Topology("cta", 1),), Layout((1,), (1,)), ("cta",))}
        if kind is MeshRegion
        else {
            "induction_var": Var(name="i", type=TensorType.scalar(DType.i64)),
            "yield_values": (),
            "extent": 2,
            "step": 1,
        }
    )
    scope = kind(
        **fields,
        params=(Var(name="param", type=value_type),),
        args=(value,),
        body=value,
        type=value_type,
    )

    with pytest.raises(VerifyError, match="region is not isolated"):
        TypeInferVisitor().visit(scope, TypeInferContext())
