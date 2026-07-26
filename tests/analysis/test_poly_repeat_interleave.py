"""``extract`` coverage for ``RepeatInterleave`` (GQA kv-head expansion) now
that it carries a registered forward ``type_relation`` (see
``tilefoundry.ir.hir.tensor.repeat_interleave``) -- before this,
``access_relation.build_relation`` returned ``None`` for it (it had no
access relation of any kind), so ``analysis.extract`` raised (its generic
path is the *only* one that consults ``type_relation_registry`` -- an op
with no registered relation has no fallback, see ``poly.py``'s
``_extract_statement``).

``y = repeat_interleave(x, repeats=4, axis=2)`` grows axis 2 from ``H`` kv
heads to ``H * repeats`` query heads. The forward relation's iteration
domain is the *output* shape (axis 2 already expanded); the output map is
identity (one write per domain point) and the input map reads
``in_idx = floor(out_idx / repeats)`` on axis 2, identity elsewhere -- so
``repeats`` consecutive output positions alias the same input element.

This test checks that shape by construction, not just non-emptiness (mirrors
``test_poly_elementwise.py``'s single-op shape, strengthened per this
task's ask): the extracted read map must equal a hand-built map expressed as
an isl existential-quantifier inequality (``repeats*o2 <= d2 <= repeats*o2 +
repeats-1``, i.e. ``o2 = floor(d2/repeats)``) -- deliberately *not* the same
``floor(a/b)`` text the relation itself emits, so this confirms the actual
semantics rather than a syntactic round-trip.

Shapes (batch=1, S=5, H=2, D=3, all pairwise distinct so an axis mixup would
show up as a mismatch) extract at plain element granularity -- this test is
about the access-map shape, not any tiling of it.
"""
from __future__ import annotations

import isl

from tilefoundry import func
from tilefoundry.analysis import TileGraph, extract
from tilefoundry.dsl import Tensor
from tilefoundry.dsl.tf import *  # noqa: F401,F403 -- repeat_interleave resolved dynamically

REPEATS = 4
B, S, H, D = 1, 5, 2, 3


@func
def gqa_expand(x: Tensor[(B, S, H, D), "f32"]) -> Tensor[(B, S, H * REPEATS, D), "f32"]:
    y = repeat_interleave(x, repeats=REPEATS, axis=2)
    return y


def test_extract_repeat_interleave_single_statement():
    """``y = repeat_interleave(x, repeats=4, axis=2)`` extracts to one
    statement instead of raising ``ExtractError``."""
    tg = extract(gqa_expand)
    assert isinstance(tg, TileGraph)
    assert len(tg.units) == 1
    unit = tg.units[0]
    assert unit.name == "RepeatInterleave"
    assert type(unit.op.target).__name__ == "RepeatInterleave"

    print("\n=== repeat_interleave: domain ===")
    print(tg.domain)
    print("=== repeat_interleave: reads ===")
    print(tg.reads)
    print("=== repeat_interleave: writes ===")
    print(tg.writes)
    print("=== repeat_interleave: deps ===")
    print(tg.deps)

    assert not tg.domain.is_empty()
    assert not tg.reads.is_empty()
    assert not tg.writes.is_empty()


def test_extract_repeat_interleave_access_maps_are_floor_div():
    """The extracted domain/reads/writes match the expected shape exactly:
    domain is the *output* iteration space; the read map divides the
    expanded axis (2) by ``repeats`` (expressed here as an existential
    inequality, independent of the relation's own ``floor(a/b)`` syntax);
    the write map is identity. No statement is self- or cross-dependent
    (single elementwise op, nothing else reads its output)."""
    tg = extract(gqa_expand)

    expected_domain = isl.set(
        f"{{ RepeatInterleave[d0,d1,d2,d3] : 0<=d0<{B} and 0<=d1<{S} "
        f"and 0<=d2<{H * REPEATS} and 0<=d3<{D} }}"
    )
    assert tg.domain.is_equal(isl.union_set("{}").union(expected_domain))

    expected_reads = isl.map(
        "{ RepeatInterleave[d0,d1,d2,d3] -> x[d0,d1,o2,d3] : "
        f"0<=d0<{B} and 0<=d1<{S} and 0<=d2<{H * REPEATS} and 0<=d3<{D} "
        f"and {REPEATS}*o2<=d2<={REPEATS}*o2+{REPEATS - 1} }}"
    )
    assert tg.reads.is_equal(isl.union_map("{}").union(expected_reads))

    expected_writes = isl.map(
        f"{{ RepeatInterleave[d0,d1,d2,d3] -> y[d0,d1,d2,d3] : "
        f"0<=d0<{B} and 0<=d1<{S} and 0<=d2<{H * REPEATS} and 0<=d3<{D} }}"
    )
    assert tg.writes.is_equal(isl.union_map("{}").union(expected_writes))

    # Elementwise, single-statement, nothing downstream reads y: no
    # self-carry (write map is injective) and no cross-statement dependence.
    assert tg.deps.is_equal(isl.union_map("{}"))
