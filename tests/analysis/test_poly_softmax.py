"""``extract`` coverage for ``SoftMax`` now that it carries a registered
forward ``type_relation`` (see ``tilefoundry.ir.hir.nn.softmax``) -- before
this, ``access_relation.build_relation`` returned ``None`` for it (only a
GLOBAL-level identity ``access_relation``, no forward relation), so
``analysis.extract`` raised (its generic path is the *only* one that
consults ``type_relation_registry`` -- an op with no registered relation has
no fallback, see ``poly.py``'s ``_extract_statement``).

SoftMax is a single fused HIR op (max/exp/sum are internal, never separate
nodes), so its access pattern is structurally identical to ``RMSNorm``'s own
registration (``rms_norm.py``'s ``_rms_norm_type_relation``): the domain is
the batch axes only (``x.shape[:-1]``) and the reduced (last) axis is an
extra existential dim on the read/write map -- one statement instance owns
an entire row, mirroring ``test_poly_model.py``'s ``RN[i]`` shape.

Shapes match ``test_poly_model.py``'s small element-granularity
convention (a batch extent of 2, a reduced extent of 64).
"""
from __future__ import annotations

import isl

from tilefoundry import func
from tilefoundry.analysis import TileGraph, extract
from tilefoundry.dsl import Tensor
from tilefoundry.dsl.tf import *  # noqa: F401,F403 -- matmul/softmax resolved dynamically


@func
def softmax_only(x: Tensor[(2, 64), "f32"]) -> Tensor[(2, 64), "f32"]:
    y = softmax(x, axis=-1)
    return y


def test_extract_softmax_single_statement():
    """``y = softmax(x, axis=-1)`` extracts to one statement: domain =
    the batch axis only, reads/writes both range over the whole
    (existentially-quantified) row -- not just the batch column --
    exactly like ``RMSNorm``'s own registration."""
    tg = extract(softmax_only)
    assert isinstance(tg, TileGraph)
    assert len(tg.units) == 1
    assert tg.units[0].name == "SoftMax"
    assert type(tg.units[0].op.target).__name__ == "SoftMax"

    print("\n=== softmax: domain ===")
    print(tg.domain)
    print("=== softmax: reads ===")
    print(tg.reads)
    print("=== softmax: writes ===")
    print(tg.writes)
    print("=== softmax: deps ===")
    print(tg.deps)

    # domain: batch axis only (extent 2), NOT batch+reduce (which would be
    # a 2-D domain like Reduce's full-element-shape relation) -- the
    # reduced axis never becomes a statement-instance dim.
    assert tg.domain.is_equal(isl.union_set("{ SoftMax[i] : 0 <= i < 2 }"))

    # reads/writes: SoftMax[i] touches x/y[i, j] for the *entire* reduced
    # axis (0<=j<64, the row's real element extent) -- i.e. one statement
    # instance genuinely reads/writes the whole row, not a per-column slice.
    expected_reads = isl.union_map(
        "{ SoftMax[i] -> x[i, j] : 0 <= i < 2 and 0 <= j < 64 }"
    )
    expected_writes = isl.union_map(
        "{ SoftMax[i] -> y[i, j] : 0 <= i < 2 and 0 <= j < 64 }"
    )
    assert tg.reads.is_equal(expected_reads)
    assert tg.writes.is_equal(expected_writes)

    # The write map is injective (distinct batch tiles never share a (i,j)
    # cell), so -- like RMSNorm's row write -- it is pure write, not also
    # folded into `reads` as a read-modify-write accumulation: buffer `y`
    # never shows up on the read side.
    assert "-> y[" not in str(tg.reads)

    # A single statement with no other statement writing `x` or reading
    # `y`: no dependence to infer.
    assert tg.deps.is_empty()


@func
def attention_scores_softmax(
    q: Tensor[(2, 2), "f32"],
    k: Tensor[(2, 2), "f32"],
    v: Tensor[(2, 2), "f32"],
) -> Tensor[(2, 2), "f32"]:
    scores = matmul(q, k)
    probs = softmax(scores, axis=-1)
    out = matmul(probs, v)
    return out


def test_extract_softmax_fuses_with_surrounding_matmuls():
    """A minimal ``scores -> softmax -> ctx`` attention fragment: SoftMax
    extracts alongside ``MatMul`` in the same ``TileGraph`` and the
    auto-inferred deps connect it on both sides (QK^T feeds softmax,
    softmax feeds the PV matmul) -- the same MM -> RN fusion shape
    ``test_poly_model.py`` validates, now with SoftMax as the row-reducing
    middle statement instead of RMSNorm."""
    tg = extract(attention_scores_softmax)
    assert isinstance(tg, TileGraph)

    names_by_op = {u.name: type(u.op.target).__name__ for u in tg.units}
    assert names_by_op == {"MM0": "MatMul", "SoftMax": "SoftMax", "MM1": "MatMul"}

    print("\n=== attention frag: domain ===")
    print(tg.domain)
    print("=== attention frag: reads ===")
    print(tg.reads)
    print("=== attention frag: writes ===")
    print(tg.writes)
    print("=== attention frag: deps (auto-inferred) ===")
    print(tg.deps)

    assert not tg.domain.is_empty()
    assert not tg.deps.is_empty()

    # Four dependences, exactly mirroring test_poly_model.py's shape (each
    # MatMul's own k-carry, plus the two cross-statement fusion edges) --
    # q/k/v are all (2,2) so every MatMul's K extent is 2 (last k-step
    # index 1, one carry step 0->1):
    #   - QK^T's k-carry: MM0[i,j,k] -> MM0[i,j,k+1]
    #   - QK^T's *last* k-step feeds every SoftMax row it overlaps (SoftMax
    #     reads the whole row -- all j positions of `scores` -- so the source
    #     is independent of j, one edge per (i,j) writer instance):
    #       MM0[i,j,1] -> SoftMax[i]
    #   - SoftMax feeds *every* (j,k) instance of the PV matmul that reads
    #     `probs[i,k]` (independent of the PV matmul's own N axis j, and for
    #     every k since SoftMax wrote the whole row in one shot):
    #       SoftMax[i] -> MM1[i,j,k]
    #   - PV's own k-carry: MM1[i,j,k] -> MM1[i,j,k+1]
    k_carry_mm0 = isl.map("{ MM0[i,j,k] -> MM0[i,j,k+1] : 0<=i<2 and 0<=j<2 and 0<=k<1 }")
    mm0_to_softmax = isl.map("{ MM0[i,j,1] -> SoftMax[i] : 0<=i<2 and 0<=j<2 }")
    softmax_to_mm1 = isl.map("{ SoftMax[i] -> MM1[i,j,k] : 0<=i<2 and 0<=j<2 and 0<=k<2 }")
    k_carry_mm1 = isl.map("{ MM1[i,j,k] -> MM1[i,j,k+1] : 0<=i<2 and 0<=j<2 and 0<=k<1 }")

    expected_total = (
        isl.union_map("{}")
        .union(k_carry_mm0)
        .union(mm0_to_softmax)
        .union(softmax_to_mm1)
        .union(k_carry_mm1)
    )
    assert tg.deps.is_equal(expected_total)
