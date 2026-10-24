"""SM80 warp-cooperative shared-memory matrix load."""

from __future__ import annotations

from tilefoundry.ir.core import Op, OpCapability
from tilefoundry.ir.core.param_def import MemoryEffect, ParamDef
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.pattern import OrPattern, SameConstraint, TensorPattern, WildcardPattern, utils
from tilefoundry.ir.tir.memory.copy import register_copy_semantics
from tilefoundry.ir.types import DType, Mesh
from tilefoundry.ir.types.storage import StorageKind as S
from tilefoundry_nvidia.cuda.ir.tir.nn.sm80_mma import Mma


@register_copy_semantics
@register_op(dialect="T", category="memory", name="ldmatrix")
class LdMatrix(Op):
    """Load one warp's shared-memory tile into the SM80 MMA A fragment."""

    capability = OpCapability("ldmatrix")
    resource = "tensor_core_engine"
    roles = ("src", "dst")
    required_execution_mesh = Mma.execution_mesh

    src = ParamDef(
        kind="input",
        effect=MemoryEffect.READ,
        pattern=TensorPattern(
            shape=(16, 16),
            dtype=WildcardPattern("dtype0", predicates=(OrPattern(DType.bf16, DType.f16),)),
            storage=S.SMEM,
            layout=utils.whole_vectors(0, (16,)),
        ),
    )
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
