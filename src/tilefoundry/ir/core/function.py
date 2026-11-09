"""The HIR ``Function``: a pure-SSA body with its callable value type."""

from __future__ import annotations

from dataclasses import MISSING, dataclass, field

from tilefoundry import services
from tilefoundry.ir import dim, expr, pattern, types


@dataclass(unsafe_hash=True)
class Function(expr.Expr):
    """Contain a pure-SSA HIR body with its callable value type.

    ``body=None`` marks a dispatch prototype. Variants and weight converters
    use tuples so function equality and hashing remain stable. Execution
    context belongs to the owning Module.

    See [hir §1.1](docs/spec/hir.md#11-function).
    """

    name: str = field(hash=False)
    params: tuple[expr.Var, ...]
    body: expr.Expr | None
    return_type: types.Type = field(hash=False)
    return_annotation: types.Type | None = field(default=None, hash=False)
    specializations: tuple[pattern.Pattern, ...] = field(default_factory=tuple)
    variants: tuple["Function", ...] = field(default_factory=tuple, hash=False)
    converters: tuple[tuple[str, "Function"], ...] = field(default_factory=tuple, hash=False)
    sealed: bool = field(default=False, compare=False, hash=False, repr=False)
    specialized_from: "Function | None" = field(
        default=None, compare=False, hash=False, repr=False
    )
    specialized_dims: tuple[tuple[str, int], ...] | None = field(
        default=None, compare=False, hash=False, repr=False
    )
    display_name: str | None = field(default=None, compare=False, hash=False, repr=False)

    def carries(self):
        return ()

    def captures(self):
        return ()

    @classmethod
    def from_parts(
        cls,
        *,
        name: str,
        params: tuple[expr.Var, ...],
        body: expr.Expr | None,
        return_type: types.Type,
        return_annotation=MISSING,
        specializations: tuple[pattern.Pattern, ...] = (),
        variants: tuple["Function", ...] = (),
        converters: tuple[tuple[str, "Function"], ...] = (),
    ) -> "Function":
        """Construct a Function whose declarations and callable type are canonical."""
        for param in params:
            param.restamp(dim.canonicalize_dims(param.annotation), dim.canonicalize_dims(param.type))
        return_annotation = dim.canonicalize_dims(
            return_type if return_annotation is MISSING else return_annotation
        )
        return_type = types.bind_annotation(dim.canonicalize_dims(return_type))
        return cls(
            name=name,
            params=params,
            body=body,
            return_type=return_type,
            return_annotation=return_annotation,
            specializations=tuple(specializations),
            variants=tuple(variants),
            converters=tuple(converters),
            type=types.callable_type_for(params, return_type),
        )

    def restamp_return(self, return_type: types.Type) -> None:
        """Replace the result type and the callable type that carries it, keeping identity."""
        self.return_type = return_type
        self.retype(types.callable_type_for(self.params, return_type))

    def relabel(self, name: str, *, display_name: str | None = None) -> None:
        """Replace the name, and the display label when one is given, keeping identity."""
        self.name = name
        if display_name is not None:
            self.display_name = display_name

    def specialize(self, declared, *, mesh=None):
        """``@base.specialize(pattern.Range(...))``: the decorator registering a shape variant."""
        return services.specialize(self, declared, mesh)

    def converter(self, weight_name: str):
        """``@base.converter(weight_name)``: the decorator that registers a per-weight converter."""
        return services.converter(self, weight_name)

    def add_variant(self, variant: "Function") -> None:
        """Append a specialization ``variant`` during authoring."""
        if self.sealed:
            raise RuntimeError(
                f"hir Function {self.name!r}: cannot add a specialization "
                f"variant after the function has entered a Module (sealed)"
            )
        self.variants = (*self.variants, variant)

    def __call__(self, *args):
        """Evaluate with one argument for every declared parameter."""
        return services.evaluate(self, *args)

    def evaluation_target(self):
        """Run me, with one argument per declared parameter and no reading."""
        return self, None

    def add_converter(self, weight_name: str, fn: "Function") -> None:
        """Register a per-weight offline converter for ``weight_name``."""
        if self.sealed:
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
        self.sealed = True
        for v in self.variants:
            v.seal()
        for _, conv in self.converters:
            conv.seal()


__all__ = ["Function"]
