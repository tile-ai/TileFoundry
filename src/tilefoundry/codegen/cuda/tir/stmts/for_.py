"""Emitter for ``tir.For`` — C-style for loop."""

from __future__ import annotations

from tilefoundry.codegen.cuda.context import CodegenContext, register_codegen_cuda
from tilefoundry.ir.core import Constant
from tilefoundry.ir.tir.stmts import For


@register_codegen_cuda(For)
def _emit(node: For, ctx: CodegenContext) -> None:
    iv_name = ctx.name_for(node.induction_var)
    for name, bound in (("start", node.start), ("stop", node.stop), ("step", node.step)):
        if not isinstance(bound, Constant):
            raise NotImplementedError(
                f"tir.For {name} is not a Constant; emitting a substituted bound would produce a silently wrong loop"
            )
    start, stop, step = node.start.value, node.stop.value, node.step.value

    ctx.emit(f"for (int {iv_name} = {start}; {iv_name} < {stop}; {iv_name} += {step}) {{")
    ctx.indent()
    ctx.emit_node(node.body)
    ctx.dedent()
    ctx.emit("}")
