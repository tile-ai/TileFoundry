"""Tag enums shared by HIR and TIR generic effect / value Ops.

These enums are owned by ``core_ir`` so that HIR and TIR can carry
the same kind value without per-dialect duplication. Lowering passes
(``HirToTirPass``) preserve the kind without remapping.
"""

from __future__ import annotations

import enum


class BinaryKind(enum.Enum):
    """Pointwise binary operation kind, shared across IRs.

    Covers arithmetic, comparison, and logical binaries; HIR sugar
    names (``add``/``mul``/``cmp_eq``/``logical_and``/...) lift to a
    single ``Binary(kind, lhs, rhs)`` form at parse time.
    """
    # Arithmetic
    ADD = "add"
    SUB = "sub"
    MUL = "mul"
    DIV = "div"
    FLOOR_DIV = "floor_div"
    MOD = "mod"
    MIN = "min"
    MAX = "max"
    # Comparison (result dtype = bool)
    EQ = "eq"
    NE = "ne"
    LT = "lt"
    LE = "le"
    GT = "gt"
    GE = "ge"
    # Logical (operands + result must be bool)
    AND = "and"
    OR = "or"


class UnaryKind(enum.Enum):
    """Pointwise unary operation kind, shared across IRs."""
    NEG = "neg"
    ABS = "abs"
    RSQRT = "rsqrt"
    CAST = "cast"
    NOT = "not"
    RELU = "relu"
    SQUARE = "square"
    EXP = "exp"
    LOG = "log"
    CEIL = "ceil"
    ROUND = "round"
    EXP2 = "exp2"
    LOG2 = "log2"


class ReduceKind(enum.Enum):
    """Reduction operation kind, shared across IRs."""
    MEAN = "mean"
    SUM = "sum"
    ABS_MAX = "abs_max"
    MAX = "max"


__all__ = ["BinaryKind", "UnaryKind", "ReduceKind"]
