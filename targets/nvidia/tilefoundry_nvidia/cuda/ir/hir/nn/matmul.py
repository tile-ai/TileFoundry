"""Where a CUDA MatMul over on-chip operands leaves its result.

A logical MatMul (gmem or undecided operands) keeps the target-neutral rule.
Once an operand is on chip, the result is what an MMA instruction the target
supports writes: each declaration admitted for MatMul is read in every
configuration it allows, and one whose A, B and C admit the operand and result
dtypes and the operand storages names the result storage by its C.
"""

from __future__ import annotations

from tilefoundry import target
from tilefoundry.interfaces import schedule, typeinfer
from tilefoundry.ir import expr, pattern, types
from tilefoundry.ir.hir import nn
from tilefoundry_nvidia.cuda import target as cuda_target

ON_CHIP = (expr.StorageKind.SMEM, expr.StorageKind.RMEM)


def mma_declarations(tgt: target.Target) -> tuple[type, ...]:
    """The atom declarations of the target's supported MatMul carriers."""
    carriers = schedule.candidate_ops(nn.MatMul)
    return tuple(
        capability.declaration
        for op_type, capability in expr.supported_op_capabilities(tgt)
        if op_type in carriers and capability.declaration is not None
    )


def matches(declared, value, configuration: dict) -> bool:
    matcher = pattern.Matcher(configuration)
    return bool(matcher.match(declared, value) and matcher.solve())


def result_storages(
    tgt: target.Target, lhs: types.TensorType, rhs: types.TensorType, dtype: expr.DType
) -> set[expr.StorageKind]:
    storages = set()
    for declaration in mma_declarations(tgt):
        for configuration in pattern.variants(declaration.parameters, vary_defaulted=True):
            a, b, c = (
                pattern.selected(getattr(declaration, role), configuration)
                for role in ("A", "B", "C")
            )
            fields = (
                (a.dtype, lhs.dtype),
                (a.storage, lhs.storage),
                (b.dtype, rhs.dtype),
                (b.storage, rhs.storage),
                (c.dtype, dtype),
            )
            if all(matches(declared, value, configuration) for declared, value in fields):
                storages.add(c.storage)
    return storages


@typeinfer.register(nn.MatMul, target=cuda_target.Target)
def cuda_matmul(call: "Call", ctx: "TypeInferContext") -> types.TensorType:
    lhs = ctx.type_of(call.args[0])
    rhs = ctx.type_of(call.args[1])
    shape, layout = nn.matmul_result_shape_and_layout(call, ctx)
    dtype = nn.matmul_result_dtype(call.target, lhs)
    if lhs.storage not in ON_CHIP and rhs.storage not in ON_CHIP:
        storage = typeinfer.resolve_anchor_storage(ctx, call, lhs.storage, rhs.storage)
        return types.TensorType(shape=shape, dtype=dtype, layout=layout, storage=storage)
    tgt = ctx.resolve_target()
    storages = result_storages(tgt, lhs, rhs, dtype)
    if not storages:
        ctx.error(
            call,
            f"no {tgt.identity} MMA reads lhs {lhs.dtype.name} {lhs.storage.name.lower()} "
            f"and rhs {rhs.dtype.name} {rhs.storage.name.lower()} into {dtype.name}; "
            "see `tilefoundry schedule facts`",
        )
    storage = storages.pop() if len(storages) == 1 else expr.StorageKind.UMAT
    return types.TensorType(shape=shape, dtype=dtype, layout=layout, storage=storage)


__all__: list[str] = []
