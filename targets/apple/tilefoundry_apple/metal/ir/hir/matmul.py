"""Metal refines result storage; core owns logical shape and shard propagation."""
from tilefoundry.ir.core.param_def import collect_param_defs
from tilefoundry.ir.hir._helpers import resolve_anchor_storage
from tilefoundry.ir.hir.nn.matmul import MatMul, matmul_result_dtype, matmul_result_shape_and_layout
from tilefoundry.ir.pattern import PatternMatcher
from tilefoundry.ir.pattern.utils import fixed_pattern_value, selected_pattern, variants
from tilefoundry.ir.types import StorageKind, TensorType
from tilefoundry.visitor_registry import register_typeinfer
from tilefoundry.visitor_registry.candidates import candidate_ops

from ...target import MetalTarget


def _result_storages(target, lhs, rhs, dtype):
    storages = set()
    for declaration in candidate_ops(MatMul, target=target):
        parameters = collect_param_defs(declaration)
        fields = {p.name: p for p in parameters}
        if not all(name in fields for name in ("lhs", "rhs", "dst")):
            continue
        attributes = tuple(p for p in parameters if p.kind == "attribute")
        for configuration in variants(attributes, vary_defaulted=True):
            a, b, c = (selected_pattern(fields[name].pattern, configuration)
                       for name in ("lhs", "rhs", "dst"))
            matcher = PatternMatcher(configuration)
            required = ((a.dtype, lhs.dtype), (a.storage, lhs.storage),
                        (b.dtype, rhs.dtype), (b.storage, rhs.storage), (c.dtype, dtype))
            if all(matcher.match(pattern, value) for pattern, value in required) and matcher.solve() is True:
                storage = fixed_pattern_value(c.storage, matcher.bindings)
                if isinstance(storage, StorageKind):
                    storages.add(storage)
    return storages


@register_typeinfer(MatMul, target=MetalTarget)
def matmul(call, ctx):
    lhs, rhs = (ctx.type_of(arg) for arg in call.args)
    shape, layout = matmul_result_shape_and_layout(call, ctx)
    dtype = matmul_result_dtype(call.target, lhs)
    on_chip = (StorageKind.SMEM, StorageKind.RMEM)
    if lhs.storage not in on_chip and rhs.storage not in on_chip:
        storage = resolve_anchor_storage(ctx, call, lhs.storage, rhs.storage)
    else:
        target = ctx.resolve_target()
        storages = _result_storages(target, lhs, rhs, dtype)
        if not storages:
            ctx.error(call, f"no {target.identity} matmul supports these operand dtypes/storages; see schedule facts")
        storage = next(iter(storages)) if len(storages) == 1 else StorageKind.UMAT
    return TensorType(shape=shape, dtype=dtype, layout=layout, storage=storage)
