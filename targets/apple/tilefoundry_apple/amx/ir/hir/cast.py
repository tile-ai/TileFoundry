"""Choose an X/Y result fragment for an exactly representable EXTR cast."""

from dataclasses import replace

from tilefoundry.ir.core.param_def import collect_param_defs
from tilefoundry.ir.hir.tensor.cast import Cast
from tilefoundry.ir.pattern import PatternMatcher
from tilefoundry.ir.pattern.utils import selected_pattern
from tilefoundry.ir.types import Layout, StorageKind
from tilefoundry.ir.types.shard_layout import ShardLayout
from tilefoundry.visitor_registry import register_typeinfer
from tilefoundry.visitor_registry.candidates import candidate_ops
from tilefoundry.visitor_registry.registries import typeinfer_registry
from tilefoundry_apple.amx.target import AmxTarget


@register_typeinfer(Cast, target=AmxTarget)
def _cast(call, ctx):
    result = typeinfer_registry.lookup(Cast)(call, ctx).type
    source = ctx.type_of(call.args[0])
    if source.storage is not StorageKind.RMEM:
        return result
    plain = replace(source, layout=source.layout.layout) if isinstance(source.layout, ShardLayout) else source
    for instruction in candidate_ops(Cast, target=ctx.target):
        for declaration in getattr(instruction, "hir_matches", ()):
            if declaration.op is not Cast:
                continue
            try:
                op = instruction(**declaration.attributes, src_dtype=source.dtype, dst_dtype=result.dtype)
            except (ValueError, TypeError):
                continue
            param = next(p for p in collect_param_defs(instruction) if p.name == 'src')
            matcher = PatternMatcher(op.bindings)
            if matcher.match(selected_pattern(param.pattern, op.bindings), plain) and matcher.solve() is True:
                layout = Layout(source.shape)
                if isinstance(source.layout, ShardLayout):
                    layout = replace(source.layout, layout=layout)
                return replace(result, layout=layout)
    return result
