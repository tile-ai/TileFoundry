"""Register HIR math names as aliases for kinded ``Binary`` and ``Unary`` Ops.

Each schema builder constructs the shared operation class with a fixed kind;
surface names do not introduce dedicated IR classes.

See [core-ir §2.3](docs/spec/core-ir.md#23-op).
"""

from __future__ import annotations

from tilefoundry.ir.core import Op
from tilefoundry.ir.core.kinds import BinaryKind, UnaryKind
from tilefoundry.ir.core.register import register_alias

from .binary import Binary
from .unary import Unary

_BINARY_ALIASES: tuple[tuple[str, BinaryKind], ...] = (
    ("add", BinaryKind.ADD),
    ("sub", BinaryKind.SUB),
    ("mul", BinaryKind.MUL),
    ("div", BinaryKind.DIV),
    ("floor_div", BinaryKind.FLOOR_DIV),
    ("mod", BinaryKind.MOD),
    ("min", BinaryKind.MIN),
    ("max", BinaryKind.MAX),
    ("minimum", BinaryKind.MIN),
    ("maximum", BinaryKind.MAX),
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


_UNARY_ALIASES: tuple[tuple[str, UnaryKind], ...] = (
    ("neg", UnaryKind.NEG),
    ("abs", UnaryKind.ABS),
    ("logical_not", UnaryKind.NOT),
    ("rsqrt", UnaryKind.RSQRT),
    ("exp", UnaryKind.EXP),
    ("log", UnaryKind.LOG),
    ("square", UnaryKind.SQUARE),
    ("ceil", UnaryKind.CEIL),
    ("round", UnaryKind.ROUND),
    ("exp2", UnaryKind.EXP2),
    ("log2", UnaryKind.LOG2),
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
