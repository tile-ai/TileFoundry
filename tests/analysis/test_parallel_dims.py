"""``TileGraph.parallel_dims`` -- per statement, per own domain dimension,
whether that dimension carries no dependence. Measured by ``extract`` off
``domain`` + ``deps`` alone, so the schedule layer reads a fact instead of
asking isl's scheduler for ``coincident``.

Asserted by op semantics over the two real qwen3-1.7B kernels rather than
against a pinned output text. A statement carries a dependence only where it
accumulates, so no statement is serial in more than one dimension, and one that
accumulates nothing is serial in none. A matmul accumulates over its own last
dimension and is always serial there. An op that reduces -- an explicit
`reduce`, a normalisation -- is serial in the reduced axis only when that axis is
a dimension of its domain at all, which is why reducing does not by itself make a
statement non-parallel.
"""
from __future__ import annotations

import isl
import pytest

from tests.models.qwen3_1_7b import decoder_layer as qwen3
from tilefoundry.analysis import extract
from tilefoundry.ir.hir.nn.matmul import MatMul

#: The ops that accumulate. Whether the accumulated axis is a dimension of the
#: statement's own domain is up to how the op builds that domain -- a matmul's
#: always is, a `reduce`'s is only when the axis survives into the domain -- so
#: what holds for all of them is that at most one dimension is serial.
_ACCUMULATING = (MatMul.__name__, "Reduce")


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
def test_only_an_accumulated_dimension_is_serial(fn):
    """No statement is serial in more than one dimension, a matmul is serial in
    its last, and a statement that accumulates nothing is serial in none.

    The count is asserted and not only the position, so a statement that came
    back serial in two dimensions cannot pass by having one of them in the place
    the accumulation was expected.
    """
    rows = _by_op(fn)
    print(f"\n=== {fn.name} parallel_dims ===")
    for name, (op, row) in rows.items():
        print(f"  {name:22s} {op:18s} {row}")

    matmuls = [name for name, (op, _) in rows.items() if op == MatMul.__name__]
    assert matmuls, "both kernels contain matmuls"
    for name in matmuls:
        op, row = rows[name]
        assert row[-1] is False, (name, op, row)
        assert all(row[:-1]), (name, op, row)

    for name, (op, row) in rows.items():
        assert row.count(False) <= 1, (name, op, row)
        if op not in _ACCUMULATING:
            assert all(row), (name, op, row)


def test_a_normalisation_is_fully_parallel_despite_reducing():
    """Named explicitly, since it is the half of the rule a pinned matmul check
    cannot cover: a normalisation reduces, and still carries no dependence on any
    dimension of its own, because the axis it reduces is not one of them."""
    rows = _by_op(qwen3.self_attention)
    norms = {name: row for name, (op, row) in rows.items() if op == "RMSNorm"}
    print("\n=== normalisations ===", norms)
    # The fused input norm, plus Qwen3's per-head q_norm and k_norm.
    assert len(norms) == 3
    assert all(all(row) for row in norms.values())
