"""Access relation handlers for HIR ``nn`` and tensor primitives."""
from __future__ import annotations

import isl

from tilefoundry.ir.core import Call, TypeInferContext, Var
from tilefoundry.ir.hir.nn.rope import RoPE
from tilefoundry.ir.hir.tensor.argmax import ArgMax
from tilefoundry.ir.hir.tensor.topk import TopK
from tilefoundry.ir.types import DType, TensorType
from tilefoundry.visitor_registry.access_relation import (
    OPAQUE,
    AccessRelations,
    access_relation_registry,
)


def _ten(shape, dtype=DType.bf16):
    return TensorType(shape=shape, dtype=dtype, layout=None, storage="gmem")


# ── RoPE ──────────────────────────────────────────────────────────────


def test_rope_relation_boundary_counts():
    q = Var(type=_ten((1, 32, 128)), name="q")
    k = Var(type=_ten((1, 4, 128)), name="k")
    cos = Var(type=_ten((4096, 128)), name="cos")
    sin = Var(type=_ten((4096, 128)), name="sin")
    pos = Var(type=_ten((1,), DType.i32), name="pos")
    call = Call(type=_ten((1, 32, 128)), target=RoPE(), args=(q, k, cos, sin, pos))
    fn = access_relation_registry.lookup(RoPE)
    rel = fn(call, TypeInferContext())
    assert isinstance(rel, AccessRelations)
    # 5 inputs: q, k, cos, sin, pos_ids.
    assert len(rel.inputs) == 5
    # 2 outputs: q_rope, k_rope.
    assert len(rel.outputs) == 2
    # q, k are identity; cos/sin/pos_ids opaque.
    assert isinstance(rel.inputs[0], isl.multi_aff)
    assert isinstance(rel.inputs[1], isl.multi_aff)
    assert rel.inputs[2] is OPAQUE
    assert rel.inputs[3] is OPAQUE
    assert rel.inputs[4] is OPAQUE
    assert isinstance(rel.outputs[0], isl.multi_aff)
    assert isinstance(rel.outputs[1], isl.multi_aff)


# ── TopK ──────────────────────────────────────────────────────────────


def test_topk_relation_input_is_axis_scan_map():
    x = Var(type=_ten((1, 128)), name="logits")
    call = Call(type=_ten((1, 128)), target=TopK(k=8), args=(x,))
    fn = access_relation_registry.lookup(TopK)
    rel = fn(call, TypeInferContext())
    assert len(rel.inputs) == 1
    assert len(rel.outputs) == 2  # values, indices
    assert isinstance(rel.inputs[0], isl.map)
    # Output identity over leading + topk dims.
    assert isinstance(rel.outputs[0], isl.multi_aff)
    assert isinstance(rel.outputs[1], isl.multi_aff)


# ── ArgMax ────────────────────────────────────────────────────────────


def test_argmax_relation_input_is_axis_scan_map():
    x = Var(type=_ten((1, 151936), DType.f32), name="logits")
    call = Call(type=_ten((1, 151936), DType.f32), target=ArgMax(), args=(x,))
    fn = access_relation_registry.lookup(ArgMax)
    rel = fn(call, TypeInferContext())
    assert len(rel.inputs) == 1
    assert len(rel.outputs) == 1
    assert isinstance(rel.inputs[0], isl.map)
    assert isinstance(rel.outputs[0], isl.multi_aff)


def test_argmax_relation_rank1_input_scalar_output():
    x = Var(type=_ten((128,), DType.f32), name="x")
    call = Call(type=_ten((128,), DType.f32), target=ArgMax(), args=(x,))
    rel = access_relation_registry.lookup(ArgMax)(call, TypeInferContext())
    assert isinstance(rel.inputs[0], isl.map)
    assert isinstance(rel.outputs[0], isl.multi_aff)
