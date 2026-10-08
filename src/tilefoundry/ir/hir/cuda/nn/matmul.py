"""Where a CUDA MatMul over on-chip operands leaves its result.

A logical MatMul (gmem or undecided operands) keeps the target-neutral rule.
Once an operand is on chip, the result is what an MMA instruction the target
supports writes: each declaration admitted for MatMul is read in every
configuration it allows, and one whose A, B and C admit the operand and result
dtypes and the operand storages names the result storage by its C.
"""

from __future__ import annotations

from tilefoundry.ir.core.op import supported_op_capabilities
from tilefoundry.ir.hir._helpers import resolve_anchor_storage
from tilefoundry.ir.hir.nn.matmul import (
    MatMul,
    matmul_result_dtype,
    matmul_result_shape_and_layout,
)
from tilefoundry.ir.pattern import PatternMatcher
from tilefoundry.ir.pattern.utils import selected_pattern, tensor_field_refusals
from tilefoundry.ir.types import DType, StorageKind, TensorType
from tilefoundry.target import CudaTarget, Target
from tilefoundry.visitor_registry import register_typeinfer

_ON_CHIP = (StorageKind.SMEM, StorageKind.RMEM)


def _mma_declarations(target: Target) -> tuple[type, ...]:
    """The atom declarations of the target's supported MatMul carriers."""
    from tilefoundry.visitor_registry.candidates import candidate_ops  # noqa: PLC0415

    carriers = candidate_ops(MatMul)
    return tuple(
        capability.declaration
        for op_type, capability in supported_op_capabilities(target)
        if op_type in carriers and capability.declaration is not None
    )


def _result_storages(
    target: Target, lhs: TensorType, rhs: TensorType, dtype: DType
) -> set[StorageKind]:
    storages = set()
    for declaration in _mma_declarations(target):
        for configuration in declaration.configurations(defaulted=True):
            a, b, c = (
                selected_pattern(getattr(declaration, role), configuration)
                for role in ("A", "B", "C")
            )
            matcher = PatternMatcher(configuration)
            if (
                not tensor_field_refusals(a, lhs, configuration)
                and not tensor_field_refusals(b, rhs, configuration)
                and matcher.match(c.dtype, dtype)
                and matcher.solve()
            ):
                storages.add(c.storage)
    return storages


@register_typeinfer(MatMul, target=CudaTarget)
def _cuda_matmul(call: "Call", ctx: "TypeInferContext") -> TensorType:
    lhs = ctx.type_of(call.args[0])
    rhs = ctx.type_of(call.args[1])
    shape, layout = matmul_result_shape_and_layout(call, ctx)
    dtype = matmul_result_dtype(call.target, lhs)
    if lhs.storage not in _ON_CHIP and rhs.storage not in _ON_CHIP:
        storage = resolve_anchor_storage(ctx, call, lhs.storage, rhs.storage)
        return TensorType(shape=shape, dtype=dtype, layout=layout, storage=storage)
    target = ctx.resolve_target()
    storages = _result_storages(target, lhs, rhs, dtype)
    if not storages:
        ctx.error(
            call,
            f"no {target.identity} MMA reads lhs {lhs.dtype.name} {lhs.storage.name.lower()} "
            f"and rhs {rhs.dtype.name} {rhs.storage.name.lower()} into {dtype.name}; "
            "see `tilefoundry schedule facts`",
        )
    storage = storages.pop() if len(storages) == 1 else StorageKind.UMAT
    return TensorType(shape=shape, dtype=dtype, layout=layout, storage=storage)
