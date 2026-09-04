"""Audit baselines for the authored TIR print/import surface."""

from __future__ import annotations

from tests._source import import_dsl
from tests.integration.test_mma_tir_handwritten import MmHandwritten
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


def test_tir_module_printer_roundtrips() -> None:
    printed = as_script(_TirBaseline)
    assert as_script(import_dsl(printed, name="_TirBaseline")) == printed


def test_handwritten_tir_mma_module_roundtrips() -> None:
    printed = as_script(MmHandwritten)
    assert as_script(import_dsl(printed, name="MmHandwritten")) == printed


def test_mixed_hir_tir_module_prints_both_function_families() -> None:
    mixed = import_dsl(
        "from tilefoundry import func, module, prim_func\n"
        "from tilefoundry.dsl import Tensor\n"
        "from tilefoundry.target import CpuTarget\n\n"
        "@module()\nclass Mixed:\n"
        "    @func\n    def h(a: Tensor[(1,), 'f32']):\n        return a\n\n"
        "    @prim_func(target=CpuTarget())\n"
        "    def t(a: Tensor[(1,), 'f32']):\n        return\n",
        name="Mixed",
    )
    printed = as_script(mixed)
    assert "@func" in printed and "@prim_func" in printed
    assert as_script(import_dsl(printed, name="Mixed")) == printed


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
    assert issubclass(Launch, Op)
    assert Launch.__name__.lower() not in schema_names
    assert not issubclass(ShapeOf, Op)
    assert not issubclass(Abort, Op)
    assert not issubclass(DispatchCall, Op)
