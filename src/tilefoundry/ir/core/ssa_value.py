"""One expression's flattened result, compared by SSA identity."""

from __future__ import annotations

from dataclasses import dataclass

from tilefoundry.ir import expr, types, utils

__all__ = ["SSAValue", "ssa_leaves"]


@dataclass(frozen=True, eq=False)
class SSAValue:
    """One expression's flattened result, compared by SSA identity."""

    expr: expr.Expr
    index: int = 0

    def __eq__(self, other):
        """Two values are equal when they are the same leaf of the same expression."""
        return isinstance(other, SSAValue) and self.expr is other.expr and self.index == other.index

    def __hash__(self):
        """Hash by the expression's identity and the leaf index."""
        return hash((id(self.expr), self.index))

    def leaf(self, held: types.Type | None = None) -> types.Type:
        """The type of this flattened result; *held* replaces the type of its expression."""
        return utils.leaves_of(self.expr.type if held is None else held)[self.index]


def ssa_leaves(node: expr.Expr) -> tuple[SSAValue, ...]:
    """Each flattened result of *expr* as an ``SSAValue``."""
    return tuple(SSAValue(node, index) for index, _ in enumerate(utils.leaves_of(node.type)))
