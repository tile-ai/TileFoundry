"""The TIR ``PrimFunction``: an effect-only function in statement position."""

from __future__ import annotations

from dataclasses import dataclass, field

from tilefoundry import services, target
from tilefoundry.ir import expr, pattern
from tilefoundry.target import cpu

from .sequential import Sequential
from .stmt import Stmt


@dataclass(unsafe_hash=True)
class PrimFunction(Stmt, expr.Annotated):
    """Contain an effect-only TIR function as a statement.

    The body is ``Sequential``. ``output_count`` records trailing output
    parameters for callable type construction.

    See [tir §2](docs/spec/tir.md#2-tir-expr-and-callable-constructs).
    """

    name: str
    params: tuple[expr.Var, ...]
    body: Sequential
    output_count: int = 1
    target: target.Target = field(default_factory=cpu.Target)
    specializations: tuple[pattern.Pattern, ...] = ()
    variants: tuple["PrimFunction", ...] = ()
    metadata: tuple[expr.IRMetadata, ...] = field(
        default=(), kw_only=True, compare=False, hash=False, repr=False
    )
    sealed: bool = field(default=False, compare=False, hash=False, repr=False)
    display_name: str | None = field(default=None, compare=False, hash=False, repr=False)

    def __post_init__(self) -> None:
        target.instance(self.target)

    def relabel(self, name: str, *, display_name: str | None = None) -> None:
        """Replace the name, and the display label when one is given, keeping identity."""
        self.name = name
        if display_name is not None:
            self.display_name = display_name

    def specialize(self, declared, *, mesh=None):
        """``@base.specialize(pattern.Range(...))``: the decorator registering a shape variant."""
        return services.specialize(self, declared, mesh)

    def add_variant(self, variant: "PrimFunction") -> None:
        if self.sealed:
            raise RuntimeError(
                f"tir PrimFunction {self.name!r}: cannot add a specialization variant after sealing"
            )
        self.variants = (*self.variants, variant)


__all__ = ["PrimFunction"]
