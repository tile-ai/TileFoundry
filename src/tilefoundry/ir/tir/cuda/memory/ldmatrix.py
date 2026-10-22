"""SM80 warp-cooperative shared-memory matrix load."""

from __future__ import annotations

from tilefoundry.ir.core import Op, OpCapability
from tilefoundry.ir.core.param_def import MemoryEffect, ParamDef
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.pattern import SameConstraint, utils
from tilefoundry.ir.tir.cuda.nn.sm80_mma import Mma
from tilefoundry.ir.tir.memory.copy import register_copy_semantics
from tilefoundry.ir.types import Mesh
from tilefoundry.ir.types.storage import StorageKind as S


@register_copy_semantics
@register_op(dialect="T", category="memory", name="ldmatrix")
class LdMatrix(Op):
    """Load one warp's shared-memory tile into the SM80 MMA A fragment."""

    capability = OpCapability("ldmatrix")
    resource = "tensor_core_engine"

    src = ParamDef(kind="input", effect=MemoryEffect.READ, pattern=utils.operand_tile(0, S.SMEM))
    dst = ParamDef(kind="input", effect=MemoryEffect.WRITE, pattern=Mma.A)
    execution_mesh = ParamDef(
        kind="attribute",
        annotation=Mesh,
        pattern=Mma.execution_mesh_pattern(),
        optional=True,
        default=None,
    )

    between = (SameConstraint("shape", "src", "dst"), SameConstraint("dtype", "src", "dst"))


__all__ = ["LdMatrix"]
