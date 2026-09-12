"""Emitter for ``tir.LetStmt``.

LetStmt wraps a value binding: `auto %var = <emit(value)>` then the scoped
body. Dispatch is two-step:

1. Look up an emitter keyed on the ``Op`` class of ``value.target`` — each
   TIR-owned Expr Op (``tir.memory.AllocTensor``, ``tir.memory.PtrOf``,
   ``tir.memory.TensorView``, etc.) registers its own emitter with
   signature ``(let: LetStmt, ctx) -> None``.
2. Recurse into the Sequential body.
"""
from __future__ import annotations

from tilefoundry.codegen.cuda.context import CudaCodegenContext
from tilefoundry.ir.core import Call
from tilefoundry.ir.tir.stmts import LetStmt
from tilefoundry.target import CudaTarget
from tilefoundry.visitor_registry.registries import Role, register_codegen


@register_codegen(CudaTarget, Role.EMIT, LetStmt)
def _emit(node: LetStmt, ctx: CudaCodegenContext) -> None:
    if not isinstance(node.value, Call):
        raise RuntimeError(
            f"LetStmt.value must be a Call (TIR-owned Expr Op), "
            f"got {type(node.value).__name__}"
        )
    op = node.value.target
    ctx.handler_for(type(op))(node, ctx)
    ctx.emit_node(node.body)
