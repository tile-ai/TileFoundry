"""``extract`` coverage for ``RoPE`` now that it carries a registered
forward ``type_relation`` (see ``tilefoundry.ir.hir.nn.rope``) -- before
this, ``access_relation.build_relation`` returned ``None`` for it (only a
GLOBAL-level ``access_relation`` with cos/sin/pos_ids marked ``OPAQUE``), so
``analysis.extract`` raised.

RoPE rotates q and k independently, and GQA gives them different head
counts (Hq != Hkv, e.g. qwen3-1.7b's 16 query / 8 key-value heads) -- so
q_rope and k_rope cannot share one iteration domain. ``extract`` now lifts
one ``RoPE`` Call into *two* statements, ``RoPE_q[b,s,Hq,d]`` and
``RoPE_k[b,s,Hkv,d]`` (``poly._rope_access``), each calling the registered relation with its own tensor paired
against itself.

The registered relation also turns cos_cache/sin_cache from opaque
(data-dependent gather) into a regular affine access: V1 assumes prefill
``pos_ids == arange(seq)``, so ``cos_cache[pos_ids[s]] == cos_cache[s]`` --
the gather degenerates to a seq-axis identity, broadcast over batch/head
(``rope[b,s,h,d] -> cos_cache[s,d]``); pos_ids itself gets the same
seq-identity access.

Shapes (batch=1, seq=4, Hq=16, Hkv=8, head_dim=128) match the task's GQA
ask, extracted at plain element granularity -- Hq/Hkv stay their own real
extents, keeping the Hq-vs-Hkv distinction this test is about.
"""
from __future__ import annotations

import isl

from tilefoundry import func
from tilefoundry.analysis import TileGraph, extract
from tilefoundry.dsl import Tensor
from tilefoundry.dsl.tf import *  # noqa: F401,F403 -- rope/repeat_interleave resolved dynamically

B, S, HQ, HKV, D, MAX_POS = 1, 4, 16, 8, 128, 8
GQA_GROUP = HQ // HKV


@func
def rope_gqa(
    q: Tensor[(B, S, HQ, D), "f32"],
    k: Tensor[(B, S, HKV, D), "f32"],
    cos_cache: Tensor[(MAX_POS, D), "f32"],
    sin_cache: Tensor[(MAX_POS, D), "f32"],
    pos_ids: Tensor[(S,), "i32"],
):
    q_rope, k_rope = rope(q, k, cos_cache, sin_cache, pos_ids)
    return q_rope, k_rope


def test_extract_rope_splits_into_q_and_k_statements():
    """One ``RoPE`` Call extracts to two statements, not one -- GQA's
    Hq != Hkv means q_rope/k_rope cannot share a domain (path A)."""
    tg = extract(rope_gqa)
    assert isinstance(tg, TileGraph)
    assert len(tg.units) == 2

    names_by_op = {u.name: type(u.op.target).__name__ for u in tg.units}
    assert names_by_op == {"RoPE_q": "RoPE", "RoPE_k": "RoPE"}


def test_extract_rope_domains_reflect_gqa_head_counts():
    """``RoPE_q``'s domain ranges over Hq=16 heads, ``RoPE_k``'s over
    Hkv=8 -- the two GQA-mismatched iteration spaces side by side."""
    tg = extract(rope_gqa)

    print("\n=== rope: domain ===")
    print(tg.domain)

    expected_q = isl.set(
        f"{{ RoPE_q[d0,d1,d2,d3] : 0<=d0<{B} and 0<=d1<{S} and 0<=d2<{HQ} and 0<=d3<{D} }}"
    )
    expected_k = isl.set(
        f"{{ RoPE_k[d0,d1,d2,d3] : 0<=d0<{B} and 0<=d1<{S} and 0<=d2<{HKV} and 0<=d3<{D} }}"
    )
    expected = isl.union_set("{}").union(expected_q).union(expected_k)
    assert tg.domain.is_equal(expected)


def test_extract_rope_cos_sin_pos_are_regular_affine_access():
    """cos_cache/sin_cache are seq+head_dim identity (batch/head
    broadcast) on *both* branches -- not OPAQUE -- under the V1 prefill
    assumption ``pos_ids == arange(seq)``; pos_ids gets the matching
    seq-identity access. Same formula for both branches: only the
    surrounding domain (Hq vs Hkv) differs."""
    tg = extract(rope_gqa)

    print("\n=== rope: reads ===")
    print(tg.reads)

    for stmt, h_extent in (("RoPE_q", HQ), ("RoPE_k", HKV)):
        bounds = f"0<=d0<{B} and 0<=d1<{S} and 0<=d2<{h_extent} and 0<=d3<{D}"
        cos_map = isl.map(f"{{ {stmt}[d0,d1,d2,d3] -> cos_cache[d1,d3] : {bounds} }}")
        sin_map = isl.map(f"{{ {stmt}[d0,d1,d2,d3] -> sin_cache[d1,d3] : {bounds} }}")
        pos_map = isl.map(f"{{ {stmt}[d0,d1,d2,d3] -> pos_ids[d1] : {bounds} }}")
        assert cos_map.is_subset(tg.reads), f"{stmt}: cos_cache access is not seq+head_dim identity"
        assert sin_map.is_subset(tg.reads), f"{stmt}: sin_cache access is not seq+head_dim identity"
        assert pos_map.is_subset(tg.reads), f"{stmt}: pos_ids access is not seq identity"


def test_extract_rope_writes_and_no_cross_branch_dependence():
    """q_rope/k_rope write distinct buffers (the same ``_{out_idx}``
    suffix convention ``_registered_access`` uses for any multi-output op),
    both writes are injective (pure elementwise, no self-read), and q/k are
    independent -- no dependence is inferred between the two branches."""
    tg = extract(rope_gqa)

    print("=== rope: writes ===")
    print(tg.writes)
    print("=== rope: deps ===")
    print(tg.deps)

    expected_write_q = isl.map(
        f"{{ RoPE_q[d0,d1,d2,d3] -> rope_0[d0,d1,d2,d3] : "
        f"0<=d0<{B} and 0<=d1<{S} and 0<=d2<{HQ} and 0<=d3<{D} }}"
    )
    expected_write_k = isl.map(
        f"{{ RoPE_k[d0,d1,d2,d3] -> rope_1[d0,d1,d2,d3] : "
        f"0<=d0<{B} and 0<=d1<{S} and 0<=d2<{HKV} and 0<=d3<{D} }}"
    )
    assert tg.writes.is_equal(isl.union_map("{}").union(expected_write_q).union(expected_write_k))
    assert tg.deps.is_equal(isl.union_map("{}"))


def test_extract_rope_k_branch_feeds_downstream_repeat_interleave():
    """End-to-end plumbing check: a GQA pipeline's
    ``repeat_interleave(k_rope, ...)`` (kv-head expansion) reads exactly
    the buffer ``RoPE_k`` wrote -- the ``TupleGetItem`` buffer-name
    passthrough (``_buffer_namer``) lines up the split statement's output
    with a real downstream consumer, and ``compute_flow`` infers the
    RoPE_k -> RepeatInterleave dependence automatically."""

    @func
    def rope_then_expand(
        q: Tensor[(B, S, HQ, D), "f32"],
        k: Tensor[(B, S, HKV, D), "f32"],
        cos_cache: Tensor[(MAX_POS, D), "f32"],
        sin_cache: Tensor[(MAX_POS, D), "f32"],
        pos_ids: Tensor[(S,), "i32"],
    ):
        q_rope, k_rope = rope(q, k, cos_cache, sin_cache, pos_ids)
        k_b = repeat_interleave(k_rope, repeats=GQA_GROUP, axis=2)
        return q_rope, k_b

    tg = extract(rope_then_expand)
    names_by_op = {u.name: type(u.op.target).__name__ for u in tg.units}
    assert names_by_op == {
        "RoPE_q": "RoPE", "RoPE_k": "RoPE", "RepeatInterleave": "RepeatInterleave",
    }

    assert "-> rope_1[" in str(tg.writes)
    assert "rope_1[" in str(tg.reads)

    k_carry = isl.map(
        f"{{ RoPE_k[d0,d1,d2,d3] -> RepeatInterleave[d0,d1,d2o,d3] : "
        f"0<=d0<{B} and 0<=d1<{S} and 0<=d2<{HKV} and 0<=d3<{D} and "
        f"{GQA_GROUP}*d2<=d2o<={GQA_GROUP}*d2+{GQA_GROUP - 1} }}"
    )
    assert k_carry.is_subset(tg.deps)
