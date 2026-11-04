"""Warp-level SM90 x4 matrix store."""

from __future__ import annotations

from tilefoundry.ir.core import Op, OpCapability
from tilefoundry.ir.core.param_def import MemoryEffect, ParamDef
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.pattern import (
    LayoutPattern,
    OrPattern,
    SameConstraint,
    ShardLayoutPattern,
    TensorPattern,
    WildcardPattern,
    utils,
)
from tilefoundry.ir.tir.memory.copy import register_copy_semantics
from tilefoundry.ir.types import DType, Layout, Mesh, Split, Topology
from tilefoundry.ir.types.storage import StorageKind as S
from tilefoundry_nvidia.cuda.ir.tir.nn.mma_atom import execution_mesh_pattern

WARP = Mesh((Topology("thread", 32),), Layout((8, 4), (4, 1)), ("lane8", "lane4"))


@register_copy_semantics
@register_op(dialect="T", category="memory", name="stmatrix")
class StMatrix(Op):
    """Store four 8x8 b16 matrices from one warp's registers."""

    capability = OpCapability("stmatrix")
    resource = "tensor_core_engine"
    roles = ("src", "dst")
    required_execution_mesh = WARP

    src = ParamDef(
        kind="input",
        effect=MemoryEffect.READ,
        pattern=TensorPattern(
            shape=(16, 16),
            dtype=OrPattern(DType.bf16, DType.f16),
            storage=S.RMEM,
            layout=ShardLayoutPattern(
                LayoutPattern((2, 8, 2, 4, 2), tuple(WildcardPattern(f"s{i}") for i in range(5))),
                (Split(1), Split(3)),
                execution_mesh_pattern(WARP),
            ),
        ),
    )
    dst = ParamDef(
        kind="input",
        effect=MemoryEffect.WRITE,
        pattern=TensorPattern(
            shape=(16, 16),
            dtype=WildcardPattern("dtype1", predicates=(OrPattern(DType.bf16, DType.f16),)),
            storage=S.SMEM,
            layout=utils.whole_vectors(1, (16,)),
        ),
    )
    execution_mesh = ParamDef(
        kind="attribute",
        annotation=Mesh,
        pattern=execution_mesh_pattern(WARP),
        optional=True,
        default=None,
    )
    between = (SameConstraint("shape", "src", "dst"), SameConstraint("dtype", "src", "dst"))

    @classmethod
    def execution_mesh_pattern(cls):
        return execution_mesh_pattern(WARP, groups=True)


__all__ = ["StMatrix"]
