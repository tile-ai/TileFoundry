"""``TileGraph.parallel_dims`` -- per statement, per own domain dimension,
whether that dimension carries no dependence. Measured by ``extract`` off
``domain`` + ``deps`` alone, so the schedule layer reads a fact instead of
asking isl's scheduler for ``coincident``.

Asserted by op semantics over the two real qwen3-1.7B kernels rather than
against a pinned output text: a matmul accumulates over its own last
dimension and so is not parallel there, and every dimension of every
elementwise / normalisation statement is (the reduction axis of RMSNorm and
SoftMax never enters the domain, so it cannot show up as a serial one).
"""
from __future__ import annotations

import isl
import pytest

from tests.models.qwen3_1_7b import decoder_layer as qwen3
from tilefoundry.analysis import extract
from tilefoundry.ir.hir.nn.matmul import MatMul


def _by_op(fn) -> dict[str, tuple[str, tuple[bool, ...]]]:
    """statement -> (op class name, its parallel_dims row)."""
    tg = extract(fn)
    ranks = {}
    sets: list["isl.set"] = []
    tg.domain.foreach_set(sets.append)
    for s in sets:
        ranks[s.get_tuple_name()] = s.dim(isl.dim_type.SET)
    out = {}
    for unit in tg.units:
        row = tg.parallel_dims[unit.name]
        assert len(row) == ranks[unit.name], (unit.name, row, ranks[unit.name])
        out[unit.name] = (type(unit.op.target).__name__, row)
    assert set(out) == set(tg.parallel_dims)
    return out


@pytest.mark.parametrize(
    "fn", [qwen3.mlp, qwen3.self_attention], ids=["mlp", "self_attention"]
)
def test_only_a_matmuls_own_reduction_dimension_is_serial(fn):
    """Every matmul's last dimension (k) is the accumulation and is not
    parallel; every other dimension of every statement is."""
    rows = _by_op(fn)
    print(f"\n=== {fn.name} parallel_dims ===")
    for name, (op, row) in rows.items():
        print(f"  {name:22s} {op:18s} {row}")

    matmuls = [name for name, (op, _) in rows.items() if op == MatMul.__name__]
    assert matmuls, "both kernels contain matmuls"
    for name in matmuls:
        _op, row = rows[name]
        assert row[-1] is False, (name, row)
        assert all(row[:-1]), (name, row)

    for name, (op, row) in rows.items():
        if op == MatMul.__name__:
            continue
        assert all(row), (name, op, row)


def test_the_reduction_free_statements_are_fully_parallel():
    """Named explicitly, since it is the half of the rule a pinned matmul
    check cannot cover: the two normalisations and the softmax carry no
    dependence at all, on any of their own dimensions."""
    rows = _by_op(qwen3.self_attention)
    reductions = {
        name: row for name, (op, row) in rows.items() if op in ("RMSNorm", "SoftMax")
    }
    print("\n=== reduction ops ===", reductions)
    assert len(reductions) == 4  # input_rms_norm + q_norm + k_norm + softmax
    assert all(all(row) for row in reductions.values())
