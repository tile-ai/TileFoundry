"""``build_schedule_tree(TileGraph) -> TileGraph`` -- the schedule tree
constructed over the polyhedral facts ``tests/analysis/test_poly_model.py``
pins, for the same minimal gemm+rmsnorm HIR ``Function``, and over the two
real qwen3-1.7B kernels: it must cover the whole domain, order every
dependence strictly, give each statement its own band, and carry
``tg.parallel_dims`` onto those bands as ``coincident``.

The legality check is the same relation-level one ``test_poly_model.py``
uses (map each dependence into time, require a lexicographically positive
delta), not a text comparison; reversing ``tg.units`` is enough to make it
fail, which is what pins it to the topological order rather than to isl.
"""
from __future__ import annotations

import dataclasses

import isl
import pytest

from tests.models.qwen3_1_7b import decoder_layer as qwen3
from tilefoundry import func
from tilefoundry.analysis import TileGraph, extract
from tilefoundry.dsl import Tensor
from tilefoundry.dsl.tf import *  # noqa: F401,F403 -- matmul/rms_norm resolved dynamically
from tilefoundry.schedule.kernel_schedule import (
    KernelScheduleError,
    band_statement,
    build_schedule_tree,
    schedule_bands,
    tile_bands,
)


@func
def gemm_rmsnorm(
    x: Tensor[(2, 4), "f32"],
    w: Tensor[(4, 2), "f32"],
    weight: Tensor[(2,), "f32"],
) -> Tensor[(2, 2), "f32"]:
    h = matmul(x, w)
    y = rms_norm(h, weight)
    return y


def _lex_positive(rank: int) -> "isl.set":
    """`{ d : first non-zero component of d is positive }`, spelled out."""
    dims = ", ".join(f"d{i}" for i in range(rank))
    pieces = []
    for k in range(rank):
        zeros = [f"d{i} = 0" for i in range(k)]
        pieces.append(f"[{dims}] : " + " and ".join(zeros + [f"d{k} > 0"]))
    return isl.set("{ " + "; ".join(pieces) + " }")


def _lex_nonpositive(deltas: "isl.union_set") -> "isl.union_set":
    """The deltas that are not lexicographically positive."""
    out = isl.union_set("{}")
    pieces: list = []
    deltas.foreach_set(pieces.append)
    for piece in pieces:
        rank = piece.dim(isl.dim_type.SET)
        out = out.union(piece.subtract(_lex_positive(rank)))
    return out


def _violations(tg: TileGraph) -> "isl.union_set":
    """The dependence deltas ``tg.tree`` does not order strictly."""
    sched_map = tg.tree.get_map()
    timed = tg.deps.apply_domain(sched_map).apply_range(sched_map)
    assert not timed.is_empty(), "every dependence must survive into time space"
    return _lex_nonpositive(timed.deltas())


def _coincident(tg: TileGraph) -> dict[str, tuple[bool, ...]]:
    return {
        band_statement(band): tuple(
            bool(band.member_get_coincident(i)) for i in range(band.n_member())
        )
        for band in schedule_bands(tg.tree)
    }


def _member_values(band: "isl.schedule_node_band", domain: "isl.union_set", pos: int) -> int:
    """How many distinct values band member ``pos`` takes over ``domain``."""
    n_member = band.n_member()
    m = band.get_partial_schedule_union_map().intersect_domain(domain)
    sets: list["isl.set"] = []
    m.range().foreach_set(sets.append)
    (box,) = sets
    keep = box.project_out(isl.dim_type.SET, pos + 1, n_member - pos - 1).project_out(
        isl.dim_type.SET, 0, pos
    )
    return int(keep.count_val().num_si())


_KERNELS = [gemm_rmsnorm, qwen3.mlp, qwen3.self_attention]
_IDS = ["gemm_rmsnorm", "mlp", "self_attention"]


@pytest.mark.parametrize("fn", _KERNELS, ids=_IDS)
def test_the_topological_tree_is_legal_and_one_band_per_statement(fn):
    """The tree covers exactly the domain, orders every dependence strictly,
    carries one band per statement in ``tg.units`` order, and marks exactly
    the members ``analysis.poly`` measured as dependence-free."""
    tg = build_schedule_tree(extract(fn))
    assert isinstance(tg, TileGraph)
    assert tg.tree.get_domain().is_equal(tg.domain)
    assert tg.ring == {}

    bands = schedule_bands(tg.tree)
    assert [band_statement(band) for band in bands] == [unit.name for unit in tg.units]
    assert _coincident(tg) == tg.parallel_dims

    violations = _violations(tg)
    print(f"\n=== {fn.name}: {len(bands)} band(s), deltas out of order: {violations}")
    assert violations.is_empty(), f"schedule violates a dependence: {violations}"


@pytest.mark.parametrize("fn", _KERNELS, ids=_IDS)
def test_reversing_the_statement_order_breaks_legality(fn):
    """The mutation that pins the legality check: ``tg.units`` is a
    dependence-respecting order, so sequencing it backwards has to put at
    least one dependence's source after its sink."""
    tg = extract(fn)
    reversed_tg = build_schedule_tree(dataclasses.replace(tg, units=tuple(reversed(tg.units))))
    violations = _violations(reversed_tg)
    print(f"\n=== {fn.name} reversed: deltas out of order: {violations}")
    assert not violations.is_empty()


def test_coincident_names_the_reduction_dimension_of_this_gemm():
    """The concrete marks, spelled out: MM's k accumulates so its own last
    member is not coincident, and RN's single member is."""
    tg = build_schedule_tree(extract(gemm_rmsnorm))
    print("\n=== coincident ===", _coincident(tg))
    assert _coincident(tg) == {"MM": (True, True, False), "RN": (True,)}


def test_tile_bands_splits_every_band_by_its_own_sizes():
    """``tile_bands`` is the whole-tree analogue of ``tile_band``: every
    statement ends up with its own tile band and point band, strided by the
    sizes decided for that statement alone."""
    tg = build_schedule_tree(extract(gemm_rmsnorm))
    tiled = tile_bands(tg.tree, {"MM": (1, 2, 4), "RN": (2,)})
    bands = schedule_bands(tiled)
    print("\n=== tiled tree ===")
    print(tiled)

    assert [band_statement(band) for band in bands] == ["MM", "MM", "RN", "RN"]
    assert [band.n_member() for band in bands] == [3, 3, 1, 1]
    # MM: extents (2, 2, 4) over tile (1, 2, 4) -> 2, 1, 1 tiles; RN: 2 over 2 -> 1.
    assert [_member_values(bands[0], tg.domain, pos) for pos in range(3)] == [2, 1, 1]
    assert [_member_values(bands[1], tg.domain, pos) for pos in range(3)] == [1, 2, 4]
    assert _member_values(bands[2], tg.domain, 0) == 1
    assert _member_values(bands[3], tg.domain, 0) == 2


def test_an_empty_tile_graph_and_a_missing_size_raise_clear_errors():
    tg = extract(gemm_rmsnorm)
    with pytest.raises(KernelScheduleError, match="tg.units is empty"):
        build_schedule_tree(dataclasses.replace(tg, units=()))

    scheduled = build_schedule_tree(tg)
    with pytest.raises(KernelScheduleError, match="no tile size decided for statement 'RN'"):
        tile_bands(scheduled.tree, {"MM": (1, 1, 1)})
