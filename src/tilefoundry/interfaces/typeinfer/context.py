"""Context and results of type inference."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NoReturn, Union

from tilefoundry.interfaces import valuation
from tilefoundry.ir import core, expr, types


@dataclass(frozen=True)
class TypeInferResults:
    """The type and optional value range produced by one inference rule."""

    type: types.Type
    value_range: tuple[int, int] | None = None


@dataclass
class TypeInferContext(valuation.Context):
    """Type inference bindings over the shared valuation environment."""

    memo: dict[int, tuple[expr.Expr, types.Type]] = field(default_factory=dict, repr=False, compare=False)
    instantiated_memo: dict[tuple, types.Type] = field(default_factory=dict, repr=False, compare=False)

    def type_of(self, node: expr.Expr) -> types.Type:
        """The type this inference derived or bound for *expr*.

        Inference reads only its own memo: a type stored on the node by an
        earlier walk is not an input, so a value this walk never reached is an
        error rather than a stale answer.
        """
        hit = self.memo.get(id(node))
        if hit is None:
            self.error(node, "this inference has no type for the value; it was not visited")
        return hit[1]

    def remember_instance(self, key: tuple, result: types.Type) -> None:
        """Keep the type one function instance inferred, so a repeat call reuses it."""
        self.instantiated_memo[key] = result

    def local_type_of(self, node: expr.Expr) -> types.Type:
        """Read an expression type without topology projection."""
        return self.type_of(node)

    def error(self, node: Union[expr.Expr, core.Stmt], msg: str) -> NoReturn:
        if isinstance(node, expr.Call):
            name = type(node.target).__name__
        else:
            name = type(node).__name__
        binding = expr.get_metadata(node, expr.BindingMetadata) if isinstance(node, expr.Expr) else None
        span = expr.get_metadata(node, expr.SourceSpanMetadata) if isinstance(node, expr.Expr) else None
        if span is not None:
            label = f" variable {binding.name!r}" if binding is not None else ""
            where = f"\n  at {span.file}:{span.line}:{span.column}{label}"
        elif isinstance(node, expr.Expr):
            location = expr.binding_name(node)
            where = f"\n  at {location}" if location else ""
        else:
            loc = getattr(node, "loc", None)
            where = f"\n  at {loc}" if loc else ""
        raise expr.VerifyError(f"{name}: {msg}{where}")


__all__ = ["TypeInferContext", "TypeInferResults"]
