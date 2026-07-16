"""Op registry contract — surface coverage + strict per-dialect resolution."""
from __future__ import annotations

import pytest

from tilefoundry.ir.core import VerifyError
from tilefoundry.ir.core.op_registry import (
    _first_schema,
    get_op_by_name,
    get_stmt_by_name,
    get_tf_by_category_name,
    iter_schemas,
)
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.stmts import For, If, LetStmt, Sequential, While
from tilefoundry.parser.dispatch import resolve_callable

# Lock list of HIR / TIR DSL names — the registry must surface every
# one. Protects against an op accidentally losing its DSL surface
# during cleanup.
#
# Kinded math sugar names (``add`` / ``sub`` / ... / ``neg``) are
# registered as **alias schemas** (no ``op_class``), so they resolve
# through ``_first_schema`` rather than ``get_op_by_name``. Listed
# separately to keep the original "real Op class" lock intact.
_HIR_REAL_OP_NAMES = frozenset({
    "argmax", "cast", "concat", "conv2d", "gather", "layer_norm", "local",
    "matmul", "quant", "rank", "relu", "reshape", "reshard", "rms_norm",
    "rope", "shape_of", "sigmoid", "slice", "softmax", "split", "stack",
    "tanh", "topk", "transpose",
})

_HIR_KINDED_ALIAS_NAMES = frozenset({
    "add", "sub", "mul", "div", "floor_div", "mod", "min", "max",
    "cmp_eq", "cmp_ne", "cmp_lt", "cmp_le", "cmp_gt", "cmp_ge",
    "logical_and", "logical_or", "neg", "abs", "logical_not",
})

_TIR_NAMES = frozenset({
    "copy", "fill", "alloc_tensor", "memory_span", "ptr_of",
    "tensor_view", "mma", "relu", "rms_norm", "reduce",
})


def test_dsl_surface_coverage_lock() -> None:
    """All HIR + TIR DSL names resolve."""
    for name in _HIR_REAL_OP_NAMES:
        assert get_op_by_name(name) is not None, f"HIR real Op {name!r} missing"
    for name in _HIR_KINDED_ALIAS_NAMES:
        s = _first_schema("tf", name)
        assert s is not None, f"HIR alias {name!r} missing"
        assert s.op_class is None, f"HIR {name!r} should be alias (op_class=None)"
    for name in _TIR_NAMES:
        assert get_stmt_by_name(name) is not None, f"TIR {name!r} missing"

    # Spot-check category-keyed view alignment with flat view.
    assert get_tf_by_category_name("nn", "rope") is get_op_by_name("rope")


def test_parser_special_forms_are_not_registered() -> None:
    """Structural Stmts (``For`` / ``If`` / ``MeshScope`` …) translate
    from Python syntax directly and must NOT appear in the schema
    registry. ``Binary`` / ``Unary`` are NOT in this list — they are
    effect-form ``Op`` subclasses registered with ``@register_op``
    (same as ``Copy`` / ``Mma`` / ``Reduce``)."""

    registered = {s.op_class for s in iter_schemas()}
    for cls in (For, If, While, LetStmt, Sequential, PrimFunction):
        assert cls not in registered, f"{cls.__name__} must not be registered"


def test_strict_per_dialect_resolution() -> None:
    """HIR-only / TIR-only names raise across dialects; trailing
    underscore effect-form is TIR-only."""

    kind, cls = resolve_callable("rope", "hir")
    assert kind == "op" and cls.name == "rope"

    kind, cls = resolve_callable("copy", "tir")
    assert kind == "stmt" and cls.name == "copy"

    with pytest.raises(VerifyError, match="unknown TIR callable 'rope'"):
        resolve_callable("rope", "tir")
    with pytest.raises(VerifyError, match="unknown HIR callable 'copy'"):
        resolve_callable("copy", "hir")

    # Effect-form selector ``copy_`` stays gated to TIR only.
    kind, cls = resolve_callable("copy_", "tir")
    assert kind == "stmt" and cls.name == "copy"
    with pytest.raises(VerifyError, match="unknown HIR callable 'copy_'"):
        resolve_callable("copy_", "hir")
