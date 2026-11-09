"""Choose an X/Y result fragment for an exactly representable EXTR cast."""

from dataclasses import replace

from tilefoundry.interfaces import schedule, typeinfer
from tilefoundry.ir import expr, pattern, types
from tilefoundry.ir.hir import tensor
from tilefoundry_apple.amx import target


@typeinfer.register(tensor.Cast, target=target.Target)
def infer_cast(call, ctx):
    result = typeinfer.registry.require(tensor.Cast)(call, ctx).type
    source = ctx.type_of(call.args[0])
    if source.storage is not expr.StorageKind.RMEM:
        return result
    plain = replace(source, layout=source.layout.layout) if isinstance(source.layout, types.ShardLayout) else source
    for instruction in schedule.candidate_ops(tensor.Cast, target=ctx.target):
        for declaration in getattr(instruction, "hir_matches", ()):
            if declaration.op is not tensor.Cast:
                continue
            try:
                op = instruction(**declaration.attributes, src_dtype=source.dtype, dst_dtype=result.dtype)
            except (ValueError, TypeError):
                continue
            param = next(p for p in expr.collect_param_defs(instruction) if p.name == 'src')
            matcher = pattern.Matcher(op.bindings)
            if matcher.match(pattern.selected(param.pattern, op.bindings), plain) and matcher.solve() is True:
                layout = types.Layout(source.shape)
                if isinstance(source.layout, types.ShardLayout):
                    layout = replace(source.layout, layout=layout)
                return replace(result, layout=layout)
    return result


__all__ = ["infer_cast"]
