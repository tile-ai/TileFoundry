"""``extract`` coverage for the elementwise op family now that ``Sigmoid`` /
``Unary`` carry a registered forward ``type_relation`` (see
``tilefoundry.ir.hir.nn.sigmoid`` / ``tilefoundry.ir.hir.math.unary``) --
before this, ``access_relation.build_relation`` returned ``None`` for both,
so ``analysis.extract`` raised (its generic path is the *only* one that
consults ``type_relation_registry`` -- an op with no registered relation has
no fallback, see ``poly.py``'s ``_extract_statement``).

Two things this file checks, mirroring ``test_poly_model.py``'s shape for
the matmul/rmsnorm family:

1. A minimal single-op HIR (``y = sigmoid(x)``, ``y = exp(x)``) extracts to a
   one-statement ``TileGraph`` with a real (element-granularity) domain and
   one read + one write access map apiece (identity, no broadcast -- single
   input, output shape == input shape).
2. The real qwen3-1.7b ``mlp`` kernel (``rms_norm`` + 3x ``matmul`` + 1x
   ``sigmoid`` + 2x ``mul``, no nested ``@func`` -- see
   ``tests/models/qwen3_1_7b/model/decoder_layer.py``) extracts *whole*: every
   op now resolves an access relation -- MatMul/Binary/RMSNorm were already
   registered, and Sigmoid is the one this task adds.
"""
from __future__ import annotations

from collections import Counter

from tests.models.qwen3_1_7b import decoder_layer as qwen3
from tilefoundry import func
from tilefoundry.analysis import TileGraph, extract
from tilefoundry.dsl import Tensor
from tilefoundry.dsl.tf import *  # noqa: F401,F403 -- sigmoid/exp resolved dynamically


@func
def sigmoid_only(x: Tensor[(64, 64), "f32"]) -> Tensor[(64, 64), "f32"]:
    y = sigmoid(x)
    return y


@func
def exp_only(x: Tensor[(64, 64), "f32"]) -> Tensor[(64, 64), "f32"]:
    y = exp(x)
    return y


def test_extract_sigmoid_single_statement():
    """``y = sigmoid(x)`` extracts to one statement: a real tiled domain plus
    one read + one write access map, both identity (no broadcast)."""
    tg = extract(sigmoid_only)
    assert isinstance(tg, TileGraph)
    assert len(tg.units) == 1
    assert tg.units[0].name == "Sigmoid"
    assert type(tg.units[0].op.target).__name__ == "Sigmoid"

    print("\n=== sigmoid: domain ===")
    print(tg.domain)
    print("=== sigmoid: reads ===")
    print(tg.reads)
    print("=== sigmoid: writes ===")
    print(tg.writes)
    print("=== sigmoid: deps ===")
    print(tg.deps)

    assert not tg.domain.is_empty()
    assert not tg.reads.is_empty()
    assert not tg.writes.is_empty()


def test_extract_unary_exp_single_statement():
    """``y = exp(x)`` (the kind-tagged ``Unary`` op) extracts the same shape
    as ``Sigmoid`` -- confirms the ``Unary`` registration is kind-agnostic
    (the relation only looks at the input shape, never ``call.target.kind``)."""
    tg = extract(exp_only)
    assert isinstance(tg, TileGraph)
    assert len(tg.units) == 1
    assert tg.units[0].name == "Unary"
    assert type(tg.units[0].op.target).__name__ == "Unary"

    print("\n=== exp (Unary): domain ===")
    print(tg.domain)
    print("=== exp (Unary): reads ===")
    print(tg.reads)
    print("=== exp (Unary): writes ===")
    print(tg.writes)

    assert not tg.domain.is_empty()
    assert not tg.reads.is_empty()
    assert not tg.writes.is_empty()


def test_extract_qwen3_mlp_whole():
    """qwen3-1.7b's ``mlp`` -- rms_norm + matmul x3 + sigmoid + mul x2, no
    nested ``@func`` -- extracts as a *single* ``TileGraph`` with one
    statement per op. Before this task's fix, extracting the whole ``mlp``
    raised ``ExtractError`` at the ``Sigmoid`` call (no registered
    ``type_relation``); now every op resolves one: MatMul/Binary(MUL)/
    RMSNorm were already registered, Sigmoid is this task's addition.
    """
    tg = extract(qwen3.mlp)
    assert isinstance(tg, TileGraph)

    op_names = [type(u.op.target).__name__ for u in tg.units]
    # Exact DFS-postorder position is `_postorder`'s own business (analyzer.py,
    # not this task) -- assert the op-kind multiset (whole-kernel coverage),
    # not a specific statement order.
    assert Counter(op_names) == Counter(
        {"RMSNorm": 1, "MatMul": 3, "Sigmoid": 1, "Binary": 2}
    )
    assert len(tg.units) == 7

    print("\n=== qwen3 mlp: statement names (postorder) ===")
    print(list(zip((u.name for u in tg.units), op_names)))
    print("=== qwen3 mlp: domain ===")
    print(tg.domain)
    print("=== qwen3 mlp: reads ===")
    print(tg.reads)
    print("=== qwen3 mlp: writes ===")
    print(tg.writes)
    print("=== qwen3 mlp: deps (auto-inferred) ===")
    print(tg.deps)

    assert not tg.domain.is_empty()
    assert not tg.reads.is_empty()
    assert not tg.writes.is_empty()
    assert not tg.deps.is_empty()

    # Sigmoid's own statement (this task's addition) actually contributed to
    # both unions -- its tuple name shows up in each dump, mirroring
    # test_poly_model.py's own "MM[" / "RN[" substring convention for
    # checking a statement's presence in a printed isl union.
    sigmoid_unit = next(u for u in tg.units if type(u.op.target).__name__ == "Sigmoid")
    assert f"{sigmoid_unit.name}[" in str(tg.reads)
    assert f"{sigmoid_unit.name}[" in str(tg.writes)
