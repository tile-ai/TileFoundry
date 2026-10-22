"""SM90 warpgroup register-fragment store through warp stmatrix instructions."""

from dataclasses import replace

from tilefoundry.evaluator.registry import register_schedule_eval, schedule_eval_registry
from tilefoundry.ir.core import Op, OpCapability
from tilefoundry.ir.core.param_def import MemoryEffect, ParamDef
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.pattern import OrPattern, SameConstraint, utils
from tilefoundry.ir.tir.cuda.nn.wgmma import Wgmma, c
from tilefoundry.ir.tir.memory.copy import Copy
from tilefoundry.ir.types import DType, Mesh
from tilefoundry.ir.types.storage import StorageKind as S
from tilefoundry.visitor_registry import register_typeinfer, register_verify_stmt
from tilefoundry.visitor_registry.access_relation import (
    identity_relations,
    register_access_relation,
)
from tilefoundry.visitor_registry.registries import typeinfer_registry, verify_stmt_registry


@register_op(dialect="T", category="memory", name="stmatrix")
class StMatrix(Op):
    """Store a 16-bit WGMMA C fragment, c/2 x4 issues per warp."""

    capability = OpCapability("stmatrix")
    resource = "tensor_core_engine"
    src = ParamDef(
        kind="input",
        effect=MemoryEffect.READ,
        pattern=replace(
            Wgmma.C,
            dtype=OrPattern(DType.bf16, DType.f16),
            predicates=(*Wgmma.C.predicates, c % 2 == 0),
        ),
    )
    dst = ParamDef(
        kind="input",
        effect=MemoryEffect.WRITE,
        pattern=utils.operand_tile(1, S.SMEM, layout=utils.whole_vectors(1, (16,))),
    )
    execution_mesh = ParamDef(
        kind="attribute",
        annotation=Mesh,
        pattern=Wgmma.execution_mesh_pattern(),
        optional=True,
        default=None,
    )
    between = (SameConstraint("shape", "src", "dst"), SameConstraint("dtype", "src", "dst"))


register_typeinfer(StMatrix)(typeinfer_registry.lookup(Copy))
register_verify_stmt(StMatrix)(verify_stmt_registry.lookup(Copy))
register_access_relation(StMatrix)(identity_relations)
register_schedule_eval(StMatrix)(schedule_eval_registry.lookup(Copy))
