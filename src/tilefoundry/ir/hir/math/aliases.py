"""HIR math surface aliases.

User-callable HIR math sugar names (``add`` / ``sub`` / ``cmp_eq`` /
``logical_and`` /
``neg`` / ...) collapse from per-name ``Op`` subclasses into a single
``Binary`` / ``Unary`` IR class with a tag-dispatched ``kind``
attribute. Each surface name registers as an **alias schema** whose
``builder`` constructs the kinded target Op directly — there is no
dedicated IR class per name.

The IR core has just ``Binary`` / ``Unary`` for kinded math, and
the surface is purely a schema-routing concern.
"""

from __future__ import annotations

from tilefoundry.ir.core import Op
from tilefoundry.ir.core.kinds import BinaryKind, UnaryKind
from tilefoundry.ir.core.register import register_alias

from .binary import Binary
from .unary import Unary

# ── Binary surface aliases ───────────────────────────────────────────────


_BINARY_ALIASES: tuple[tuple[str, BinaryKind], ...] = (
    ("add", BinaryKind.ADD),
    ("sub", BinaryKind.SUB),
    ("mul", BinaryKind.MUL),
    ("div", BinaryKind.DIV),
    ("floor_div", BinaryKind.FLOOR_DIV),
    ("mod", BinaryKind.MOD),
    ("min", BinaryKind.MIN),
    ("max", BinaryKind.MAX),
    ("cmp_eq", BinaryKind.EQ),
    ("cmp_ne", BinaryKind.NE),
    ("cmp_lt", BinaryKind.LT),
    ("cmp_le", BinaryKind.LE),
    ("cmp_gt", BinaryKind.GT),
    ("cmp_ge", BinaryKind.GE),
    ("logical_and", BinaryKind.AND),
    ("logical_or", BinaryKind.OR),
)


def _make_binary_alias(name: str, kind: BinaryKind) -> None:
    @register_alias(
        dialect="tf",
        category="math",
        name=name,
        params=[Binary.lhs, Binary.rhs],
    )
    def _alias(_kind: BinaryKind = kind) -> Op:
        return Binary(kind=_kind)

    _alias.__name__ = f"_{name}_alias"
    _alias.__qualname__ = f"tilefoundry.ir.hir.math.aliases.{_alias.__name__}"


for _n, _k in _BINARY_ALIASES:
    _make_binary_alias(_n, _k)


# ── Unary surface aliases ────────────────────────────────────────────────


_UNARY_ALIASES: tuple[tuple[str, UnaryKind], ...] = (
    ("neg", UnaryKind.NEG),
    ("abs", UnaryKind.ABS),
    ("logical_not", UnaryKind.NOT),
    ("rsqrt", UnaryKind.RSQRT),
    # SQUARE is a first-class UnaryKind, not a parser-level expansion
    # to mul(x, x), so codegen / lowering can see the "this is a
    # square" intent explicitly.
    ("square", UnaryKind.SQUARE),
)


def _make_unary_alias(name: str, kind: UnaryKind) -> None:
    @register_alias(
        dialect="tf",
        category="math",
        name=name,
        params=[Unary.x],
    )
    def _alias(_kind: UnaryKind = kind) -> Op:
        return Unary(kind=_kind)

    _alias.__name__ = f"_{name}_alias"
    _alias.__qualname__ = f"tilefoundry.ir.hir.math.aliases.{_alias.__name__}"


for _n, _k in _UNARY_ALIASES:
    _make_unary_alias(_n, _k)


__all__: list[str] = []
