from __future__ import annotations

from dataclasses import dataclass, field

from tilefoundry.ir.core.errors import VerifyError
from tilefoundry.ir.core.metadata import IRMetadata
from tilefoundry.ir.core.op import Op

from ..types.tensor_type import Type


@dataclass(frozen=True)
class Expr:
    """Typed SSA value. Base of all expression nodes (hir + tir-embedded).

    `type` is the Expr's result type (TensorType for single-output, TupleType
    for multi-output); `source` is optional debug info. Both kw-only so
    subclasses can declare positional fields without default-order clashes.
    """
    type: Type = field(kw_only=True)
    loc: str | None = field(default=None, kw_only=True)
    metadata: tuple[IRMetadata, ...] = field(
        default_factory=tuple,
        kw_only=True,
        compare=False,
        hash=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        seen: set[type[IRMetadata]] = set()
        for value in self.metadata:
            if not isinstance(value, IRMetadata):
                where = f"\n  at {self.loc}" if self.loc else ""
                raise VerifyError(
                    f"{type(self).__name__} metadata entries must be IRMetadata, "
                    f"got {type(value).__name__}{where}"
                )
            value_cls = type(value)
            if value_cls in seen:
                where = f"\n  at {self.loc}" if self.loc else ""
                raise VerifyError(
                    f"{type(self).__name__} has duplicate {value_cls.__name__} "
                    f"metadata{where}"
                )
            seen.add(value_cls)


@dataclass(frozen=True)
class Var(Expr):
    name: str
    is_const: bool = False


@dataclass(frozen=True)
class Constant(Expr):
    value: object


@dataclass(frozen=True)
class Call(Expr):
    """Call to an Op. Produces a value. Cannot be top-level Stmt in tir (§8.5)."""
    target: Op
    args: tuple[Expr, ...]


@dataclass(frozen=True)
class Tuple(Expr):
    """Value-form explicit tuple construction.

    ``Tuple((a, b))``: the type is ``TupleType(fields=(a.type, b.type))``. Not
    a registered Op — an IR-level construct emitted by the parser for
    ``return (a, b)`` bodies and per-axis scalar tuples (e.g. ``insert_slice``
    offsets).
    """
    elements: tuple[Expr, ...]


