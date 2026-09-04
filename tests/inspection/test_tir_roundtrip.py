"""Audit baselines for the authored TIR print/import surface."""

from __future__ import annotations

import pytest

from tilefoundry import module, prim_func
from tilefoundry.dsl import Tensor
from tilefoundry.inspection import as_script
from tilefoundry.ir.core import Op
from tilefoundry.ir.core.op_registry import get_stmt_by_name, iter_schema_names
from tilefoundry.ir.tir.dispatch import DispatchCall
from tilefoundry.ir.tir.launch import Launch
from tilefoundry.ir.tir.shape import ShapeOf
from tilefoundry.ir.tir.stmts import Abort
from tilefoundry.ir.types import TensorType, UnitType
from tilefoundry.target import CpuTarget
from tilefoundry.visitor_registry import typeinfer_registry

TIR_SUPPORT_MATRIX = {
    "alloc_tensor": "value-form",
    "binary": "effect-form",
    "clamp": "effect-form",
    "copy": "effect-form",
    "copy_async": "effect-form",
    "cp_async_commit": "effect-form",
    "cp_async_wait": "effect-form",
    "fill": "effect-form",
    "memory_span": "value-form",
    "mma": "effect-form",
    "ptr_of": "value-form",
    "reduce": "effect-form",
    "relu": "effect-form",
    "rms_norm": "effect-form",
    "sync": "effect-form",
    "tensor_view": "value-form",
    "unary": "effect-form",
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

    inferred_forms = {}
    for name, expected in TIR_SUPPORT_MATRIX.items():
        op_cls = get_stmt_by_name(name)
        assert op_cls is not None, name
        infer = typeinfer_registry.lookup(op_cls)
        assert infer is not None, name
        result_type = infer.__annotations__["return"]
        if isinstance(result_type, str):
            result_type = result_type.rsplit(".", 1)[-1]
        inferred_forms[name] = (
            "effect-form" if result_type in (UnitType, "UnitType") else "value-form"
            if result_type in (TensorType, "TensorType")
            else None
        )
        assert inferred_forms[name] == expected


def test_unregistered_printable_tir_nodes_are_named() -> None:
    schema_names = set(iter_schema_names("T"))
    assert UNREGISTERED_PRINTABLE_TIR_NODES == {Launch, ShapeOf, Abort, DispatchCall}
    assert issubclass(Launch, Op)
    assert Launch.__name__.lower() not in schema_names
    assert not issubclass(ShapeOf, Op)
    assert not issubclass(Abort, Op)
    assert not issubclass(DispatchCall, Op)
