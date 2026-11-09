"""Warp-level SM90 x4 matrix store."""

from __future__ import annotations

from functools import partial

from tilefoundry.ir import expr, pattern, types
from tilefoundry.ir.tir import memory
from tilefoundry_nvidia.cuda.ir.tir import nn

WARP = types.Mesh((types.Topology("thread", 32),), types.Layout((8, 4), (4, 1)), ("lane8", "lane4"))


@memory.register_copy_semantics
@expr.register_op(dialect="T", category="memory", name="stmatrix")
class StMatrix(expr.Op):
    """Store four 8x8 b16 matrices from one warp's registers."""

    capability = expr.OpCapability("stmatrix")
    resource = "tensor_core_engine"
    roles = ("src", "dst")
    required_execution_mesh = WARP

    src = expr.ParamDef(
        kind="input",
        effect=expr.MemoryEffect.READ,
        pattern=pattern.Tensor(
            shape=(16, 16),
            dtype=pattern.Or(expr.DType.bf16, expr.DType.f16),
            storage=expr.StorageKind.RMEM,
            layout=pattern.ShardLayout(
                pattern.Layout((2, 8, 2, 4, 2), tuple(pattern.Wildcard(f"s{i}") for i in range(5))),
                (types.Split(1), types.Split(3)),
                nn.execution_mesh_pattern(WARP),
            ),
        ),
    )
    dst = expr.ParamDef(
        kind="input",
        effect=expr.MemoryEffect.WRITE,
        pattern=pattern.Tensor(
            shape=(16, 16),
            dtype=pattern.Wildcard("dtype1", predicates=(pattern.Or(expr.DType.bf16, expr.DType.f16),)),
            storage=expr.StorageKind.SMEM,
            layout=pattern.whole_vectors(1, (16,)),
        ),
    )
    execution_mesh = expr.ParamDef(
        kind="attribute",
        annotation=types.Mesh,
        pattern=nn.execution_mesh_pattern(WARP),
        optional=True,
        default=None,
    )
    between = (pattern.SameConstraint("shape", "src", "dst"), pattern.SameConstraint("dtype", "src", "dst"))

    execution_mesh_pattern = staticmethod(partial(nn.execution_mesh_pattern, WARP, groups=True))


__all__ = ["StMatrix"]
