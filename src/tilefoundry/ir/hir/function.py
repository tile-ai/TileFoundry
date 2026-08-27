from __future__ import annotations

from dataclasses import dataclass, field

from tilefoundry.ir.core import Expr, Var
from tilefoundry.ir.core.pattern import Pattern
from tilefoundry.ir.types import Type, callable_type_for
from tilefoundry.ir.types.substitute import canonicalize_dims


@dataclass(unsafe_hash=True)
class Function(Expr):
    """Contain a pure-SSA HIR body with its callable value type.

    ``body=None`` marks a dispatch prototype. Variants and weight converters
    use tuples so function equality and hashing remain stable. Execution
    context belongs to the owning Module.

    See [hir §1.1](docs/spec/hir.md#11-function).
    """

    name: str = field(hash=False)
    params: tuple[Var, ...]
    body: Expr | None
    return_type: Type = field(hash=False)
    specializations: tuple[Pattern, ...] = field(default_factory=tuple)
    variants: tuple["Function", ...] = field(default_factory=tuple, hash=False)
    converters: tuple[tuple[str, "Function"], ...] = field(default_factory=tuple, hash=False)
    _sealed: bool = field(default=False, compare=False, hash=False, repr=False)
    _specialized_from: "Function | None" = field(
        default=None, compare=False, hash=False, repr=False
    )
    _specialized_dims: tuple[tuple[str, int], ...] | None = field(
        default=None, compare=False, hash=False, repr=False
    )
    _display_name: str | None = field(
        default=None, compare=False, hash=False, repr=False
    )

    @classmethod
    def build(
        cls,
        *,
        name: str,
        params: tuple[Var, ...],
        body: Expr | None,
        return_type: Type,
        specializations: tuple[Pattern, ...] = (),
        variants: tuple["Function", ...] = (),
        converters: tuple[tuple[str, "Function"], ...] = (),
    ) -> "Function":
        """Construct a Function whose declarations and callable type are canonical."""
        for param in params:
            canonical = canonicalize_dims(param.type)
            if canonical is not param.type:
                param.type = canonical
        return_type = canonicalize_dims(return_type)
        return cls(
            name=name,
            params=params,
            body=body,
            return_type=return_type,
            specializations=tuple(specializations),
            variants=tuple(variants),
            converters=tuple(converters),
            type=callable_type_for(params, return_type),
        )

    def add_variant(self, variant: "Function") -> None:
        """Append a specialization ``variant`` during authoring."""
        if getattr(self, "_sealed", False):
            raise RuntimeError(
                f"hir Function {self.name!r}: cannot add a specialization "
                f"variant after the function has entered a Module (sealed)"
            )
        self.variants = (*self.variants, variant)

    def add_converter(self, weight_name: str, fn: "Function") -> None:
        """Register a per-weight offline converter for ``weight_name``."""
        if getattr(self, "_sealed", False):
            raise RuntimeError(
                f"hir Function {self.name!r}: cannot add a converter after "
                f"the function has entered a Module (sealed)"
            )
        if any(existing == weight_name for existing, _ in self.converters):
            raise ValueError(
                f"hir Function {self.name!r}: a converter for weight "
                f"{weight_name!r} is already registered"
            )
        self.converters = (*self.converters, (weight_name, fn))

    def seal(self) -> None:
        """Freeze authoring mutation: ``add_variant`` raises afterwards."""
        self._sealed = True
        for v in self.variants:
            v.seal()
        for _, conv in self.converters:
            conv.seal()


__all__ = ["Function"]
