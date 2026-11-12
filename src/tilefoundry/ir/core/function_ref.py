"""FunctionRef — the static name of a callee in a Module tree."""

from __future__ import annotations

from dataclasses import dataclass, field

from tilefoundry.ir import expr


@dataclass(unsafe_hash=True)
class FunctionRef(expr.Expr):
    """Leaf ``Expr`` naming a callee by its slot path and function name.

    A ``str`` segment enters a child slot, an ``int`` segment selects one
    instance of an indexed slot, and ``None`` marks an index that the call
    supplies as a leading operand. ``name=None`` names the entry of the
    Module the path reaches. The reference holds no Module and no type: the
    reader supplies the Module it resolves from.
    """

    path: tuple[str | int | None, ...] = ()
    name: str | None = None
    type: None = field(default=None, kw_only=True, compare=False, hash=False)

    def indices(self, args):
        """The leading operands of a call that are run-time instance indices."""
        return args[: self.path.count(None)]

    def arguments(self, args):
        """The operands of a call that align with the callee's parameters."""
        return args[self.path.count(None) :]

    @property
    def qualified_name(self) -> str:
        """The dotted spelling, such as ``layers.0.run``, used by C symbols and runtime names."""
        return ".".join(str(part) for part in (*self.path, self.name) if part is not None)


__all__ = ["FunctionRef"]
