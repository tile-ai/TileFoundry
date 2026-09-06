from __future__ import annotations

from dataclasses import dataclass, field

from tilefoundry.ir.core import Var
from tilefoundry.ir.core.pattern import Pattern
from tilefoundry.ir.tir.stmt import Stmt
from tilefoundry.ir.tir.stmts import Sequential
from tilefoundry.target.base import Target, target_instance


def _default_target():
    """Construct the compiler-owned default after the IR import graph is ready."""
    from tilefoundry.target import default_target  # noqa: PLC0415

    return default_target()


@dataclass(frozen=True)
class PrimFunction(Stmt):
    """Contain an effect-only TIR function as a statement.

    The body is ``Sequential``. ``output_count`` records trailing output
    parameters for callable type construction.

    See [tir §2](docs/spec/tir.md#2-tir-expr-and-callable-constructs).
    """

    name: str
    params: tuple[Var, ...]
    body: Sequential
    output_count: int = 1
    target: Target = field(default_factory=_default_target)
    specializations: tuple[Pattern, ...] = ()
    variants: tuple["PrimFunction", ...] = ()

    def __post_init__(self) -> None:
        target_instance(self.target)

    def add_variant(self, variant: "PrimFunction") -> None:
        if getattr(self, "_sealed", False):
            raise RuntimeError(f"tir PrimFunction {self.name!r}: cannot add a specialization variant after sealing")
        if variant.name == self.name and len(variant.specializations) == 1:
            pat = variant.specializations[0]
            if hasattr(pat, "dim_var"):
                object.__setattr__(variant, "name", f"{self.name}${pat.dim_var}${pat.lo}_{pat.hi}")
        object.__setattr__(self, "variants", (*self.variants, variant))


__all__ = ["PrimFunction"]
