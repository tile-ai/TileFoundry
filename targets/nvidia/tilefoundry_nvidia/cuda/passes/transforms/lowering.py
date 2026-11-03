"""Lower scheduled HIR for CUDA with the ``sync`` steps before each node that has them.

See [passes §7.1](targets/nvidia/tilefoundry_nvidia/cuda/docs/spec/passes.md#71-converthirtotir).
"""

from __future__ import annotations

from tilefoundry.analysis.sync import SyncMetadata, SyncPlanMetadata
from tilefoundry.ir.core import Expr, get_metadata
from tilefoundry.ir.tir import CpAsyncCommit, CpAsyncWait
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.stmts import Evaluate
from tilefoundry.ir.tir.sync import Sync
from tilefoundry.passes.transforms.convert_hir_to_tir import (
    ConvertHIRToTIR,
    Lowering,
    LoweringError,
)
from tilefoundry.target import bind
from tilefoundry_nvidia.cuda import CudaTarget


class CudaLowering(Lowering):
    """The core lowering, emitting the steps the ``sync`` analysis attached to each node."""

    def run(self) -> PrimFunction:
        function = self.context.function
        if get_metadata(function, SyncPlanMetadata) is None:
            raise LoweringError(
                f"{function.name!r}: needs the result of analyze(..., analysis=('sync',))"
            )
        return super().run()

    def before(self, owner: Expr, ctx, tag=None) -> None:
        record = get_metadata(owner, SyncMetadata)
        for step in () if record is None else record.at(tag):
            if step.kind == "commit":
                ctx.builder.add(Evaluate(CpAsyncCommit(), ()))
            elif step.kind == "wait":
                ctx.builder.add(Evaluate(CpAsyncWait(n=0), ()))
            else:
                ctx.builder.add(Evaluate(Sync(mesh=step.mesh), ()))


@bind(CudaTarget, ConvertHIRToTIR)
class CudaConvertHIRToTIR(ConvertHIRToTIR):
    """Lower scheduled HIR with the visitor that emits the sync steps."""

    visitor = CudaLowering


__all__ = ["CudaConvertHIRToTIR", "CudaLowering"]
