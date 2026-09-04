"""Audit baselines for the authored TIR print/import surface."""

from __future__ import annotations

import pytest

from tilefoundry import module, prim_func
from tilefoundry.dsl import Tensor
from tilefoundry.inspection import as_script
from tilefoundry.ir.core.op_registry import iter_schema_names
from tilefoundry.ir.tir.dispatch import DispatchCall
from tilefoundry.ir.tir.launch import Launch
from tilefoundry.ir.tir.shape import ShapeOf
from tilefoundry.ir.tir.stmts import Abort
from tilefoundry.target import CpuTarget

TIR_SUPPORT_MATRIX = {
    "alloc_tensor": "value-form",
    "binary": "value-form",
    "clamp": "value-form",
    "copy": "effect-form",
    "copy_async": "effect-form",
    "cp_async_commit": "effect-form",
    "cp_async_wait": "effect-form",
    "fill": "effect-form",
    "memory_span": "value-form",
    "mma": "effect-form",
    "ptr_of": "value-form",
    "reduce": "value-form",
    "relu": "value-form",
    "rms_norm": "value-form",
    "sync": "effect-form",
    "tensor_view": "value-form",
    "unary": "value-form",
}

UNREGISTERED_PRINTABLE_TIR_NODES = {
    Launch,
    ShapeOf,
    Abort,
    DispatchCall,
}


@module(entry="tir_entry")
class _TirBaseline:
    @prim_func(target=CpuTarget())
    def tir_entry(a: Tensor[(1,), "f32"]):
        return


@pytest.mark.xfail(strict=True, raises=TypeError, reason="TIR module printer is added in M1")
def test_as_script_cannot_serialize_a_tir_module_today() -> None:
    as_script(_TirBaseline)


def test_t_schema_inventory_matches_the_tir_support_matrix() -> None:
    assert sorted(iter_schema_names("T")) == sorted(TIR_SUPPORT_MATRIX)
    assert set(TIR_SUPPORT_MATRIX.values()) == {"value-form", "effect-form"}


def test_unregistered_printable_tir_nodes_are_named() -> None:
    assert {Launch, ShapeOf, Abort, DispatchCall} == UNREGISTERED_PRINTABLE_TIR_NODES
