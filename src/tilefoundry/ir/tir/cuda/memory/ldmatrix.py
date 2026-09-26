"""SM80 warp-cooperative shared-memory matrix load."""

from __future__ import annotations

from tilefoundry.ir.core import Op
from tilefoundry.ir.core.param_def import MemoryEffect, ParamDef
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.pattern import utils
from tilefoundry.ir.tir.cuda.nn.sm80_mma import Mma
from tilefoundry.ir.tir.memory.copy import Copy
from tilefoundry.ir.types import Mesh
from tilefoundry.ir.types.storage import StorageKind as S
from tilefoundry.visitor_registry import register_typeinfer, register_verify_stmt
from tilefoundry.visitor_registry.registries import typeinfer_registry, verify_stmt_registry


@register_op(dialect="T", category="memory", name="ldmatrix")
class LdMatrix(Op):
    """Load one warp's shared-memory tile into the SM80 MMA A fragment."""

    capability = "tensor_core"

    src = ParamDef(
        kind="input", effect=MemoryEffect.READ, pattern=utils.operand_tile(0, S.SMEM)
    )
    dst = ParamDef(kind="input", effect=MemoryEffect.WRITE, pattern=Mma.A)
    scope = ParamDef(
        kind="attribute",
        annotation=Mesh,
        pattern=Mma.scope_pattern(),
        optional=True,
        default=None,
    )


register_typeinfer(LdMatrix)(typeinfer_registry.lookup(Copy))
register_verify_stmt(LdMatrix)(verify_stmt_registry.lookup(Copy))


__all__ = ["LdMatrix"]
