"""SM80 warp-cooperative shared-memory matrix load."""

from __future__ import annotations

from tilefoundry.ir import expr, pattern, types
from tilefoundry.ir.tir import memory
from tilefoundry_nvidia.cuda.ir.tir import nn


@memory.register_copy_semantics
@expr.register_op(dialect="T", category="memory", name="ldmatrix")
class LdMatrix(expr.Op):
    """Load one warp's shared-memory tile into the SM80 MMA A fragment."""

    capability = expr.OpCapability("ldmatrix")
    resource = "tensor_core_engine"
    roles = ("src", "dst")
    required_execution_mesh = nn.Mma.execution_mesh

    src = expr.ParamDef(
        kind="input",
        effect=expr.MemoryEffect.READ,
        pattern=pattern.Tensor(
            shape=(16, 16),
            dtype=pattern.Wildcard("dtype0", predicates=(pattern.Or(expr.DType.bf16, expr.DType.f16),)),
            storage=expr.StorageKind.SMEM,
            layout=pattern.whole_vectors(0, (16,)),
        ),
    )
    dst = expr.ParamDef(kind="input", effect=expr.MemoryEffect.WRITE, pattern=nn.Mma.A)
    execution_mesh = expr.ParamDef(
        kind="attribute",
        annotation=types.Mesh,
        pattern=nn.Mma.execution_mesh_pattern(),
        optional=True,
        default=None,
    )

    between = (pattern.SameConstraint("shape", "src", "dst"), pattern.SameConstraint("dtype", "src", "dst"))


__all__ = ["LdMatrix"]
