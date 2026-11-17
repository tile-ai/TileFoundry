"""What one reverse-mode rule may read, and how it builds a new call."""

from __future__ import annotations

from dataclasses import dataclass

from tilefoundry.interfaces import typeinfer
from tilefoundry.ir import expr, types
from tilefoundry.ir.expr import op_registry

__all__ = ["Context"]


@dataclass
class Context(typeinfer.TypeInferContext):
    """Type inference over the forward, plus construction of typed calls.

    A rule names the Ops it builds by their DSL name, so one Op file never
    imports another: the rule of Binary builds a reduce, and the rule of
    Reduce builds a binary.
    """

    def call(self, name: str, *args: expr.Expr, **attrs) -> expr.Call:
        """Build ``tf.<name>(*args, **attrs)`` through the op registry and infer its type."""
        schema = op_registry.lookup("tf", (), name)
        if not isinstance(schema, expr.OpSchema):
            raise ValueError(f"autodiff: no tf op named {name!r}")
        call = expr.Call(
            type=types.TensorType.from_scalar(expr.DType.f32),
            target=schema.builder(**attrs),
            args=args,
        )
        typeinfer.TypeInferVisitor().visit(call, self)
        return call
