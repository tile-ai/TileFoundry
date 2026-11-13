"""FunctionRef — the static name of a callee in a Module tree."""

from __future__ import annotations

from dataclasses import dataclass, field

from tilefoundry.ir import expr, types


def is_index_type(index_type) -> bool:
    """Whether a run-time instance index of this type is admitted: an integer scalar."""
    return (
        isinstance(index_type, types.TensorType)
        and index_type.shape == ()
        and isinstance(index_type.dtype, expr.IntegerDType)
    )


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

    def check_indices(self, index_types) -> None:
        """Require one integer scalar for each run-time index of the path."""
        slots = tuple(self.path[k - 1] for k, segment in enumerate(self.path) if segment is None)
        if len(index_types) != len(slots):
            raise ValueError(
                f"symbol {self.qualified_name!r}: expected {len(slots)} index operand(s), "
                f"got {len(index_types)}"
            )
        for slot, index_type in zip(slots, index_types):
            if not is_index_type(index_type):
                raise ValueError(f"ModuleList slot {slot!r}: index must be an integer scalar")

    @property
    def qualified_name(self) -> str:
        """The dotted spelling, such as ``layers.0.run``, used by C symbols and runtime names.

        A run-time index prints as ``*``; an entry reference ends at its path.
        """
        parts = ("*" if part is None else str(part) for part in self.path)
        return ".".join((*parts, *(() if self.name is None else (self.name,))))


__all__ = ["FunctionRef", "is_index_type"]
