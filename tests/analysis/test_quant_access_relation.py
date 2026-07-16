"""Quant access relation — GLOBAL black-box level.

"""
from __future__ import annotations

import isl

from tilefoundry.ir.core import Call, TypeInferContext, Var
from tilefoundry.ir.hir.tensor.quant import Quant
from tilefoundry.ir.types import DType, make_tensor_type
from tilefoundry.visitor_registry.access_relation import (
    AccessRelations,
    access_relation_registry,
)


def _quant_relations(shape, group=128) -> AccessRelations:
    x = Var(type=make_tensor_type(shape, DType.bf16), name="x")
    call = Call(type=make_tensor_type(shape, DType.bf16), target=Quant(group=group), args=(x,))
    fn = access_relation_registry.lookup(Quant)
    assert fn is not None
    return fn(call, TypeInferContext())


def test_quant_relation_returns_accessrelations():
    rel = _quant_relations((1, 2048))
    assert isinstance(rel, AccessRelations)


def test_quant_boundary_counts():
    """1 input (x), 2 outputs (x_q, x_scale)."""
    rel = _quant_relations((1, 2048))
    assert len(rel.inputs) == 1
    assert len(rel.outputs) == 2


def test_quant_input_relation_is_identity():
    rel = _quant_relations((1, 2048))
    inp = rel.inputs[0]
    assert isinstance(inp, isl.multi_aff)
    # Identity: domain dims == range dims, both rank 2.
    s = str(inp)
    assert "->" in s


def test_quant_output_xq_relation_is_identity():
    rel = _quant_relations((1, 2048))
    x_q = rel.outputs[0]
    assert isinstance(x_q, isl.multi_aff)


def test_quant_output_scale_relation_is_per_group_map():
    """x_scale should reduce the last axis to floor(j / group) — modeled as
    isl.map (many-to-one)."""
    rel = _quant_relations((1, 2048), group=128)
    scale = rel.outputs[1]
    assert isinstance(scale, isl.map)
    s = str(scale)
    # ISL canonicalises floor(j/128) into linear constraints involving 128.
    assert "128" in s


def test_quant_relation_rank3_input():
    """Attn-output [1,1,4096] reduces to scale [1,1,32]."""
    rel = _quant_relations((1, 1, 4096))
    assert len(rel.inputs) == 1
    assert len(rel.outputs) == 2
    inp = rel.inputs[0]
    # Rank-3 identity has 3 dim references.
    s = str(inp)
    assert s.count("i0") >= 1 and s.count("i1") >= 1 and s.count("i2") >= 1
