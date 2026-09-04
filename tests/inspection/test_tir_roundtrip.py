"""Audit baselines for the authored TIR print/import surface."""

from __future__ import annotations

import subprocess

import pytest

import tilefoundry.codegen.cuda  # noqa: F401
from tests._source import import_dsl
from tests.integration.test_mma_tir_handwritten import MmHandwritten
from tests.ir.test_dispatch_call import _build_module as build_dispatch_functions
from tests.ir.tir.test_async_copy import AsyncStage
from tests.ir.tir.test_sync import SyncSquare
from tilefoundry import module, prim_func
from tilefoundry.codegen.cuda.context import CodegenContext
from tilefoundry.dsl import T, Tensor
from tilefoundry.inspection import as_script
from tilefoundry.ir.core import Constant, Op, Var
from tilefoundry.ir.core.kinds import BinaryKind, ReduceKind, UnaryKind
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.core.op_registry import get_stmt_by_name, iter_schema_names
from tilefoundry.ir.tir.dispatch import DispatchCall
from tilefoundry.ir.tir.launch import Launch
from tilefoundry.ir.tir.shape import ShapeOf
from tilefoundry.ir.tir.stmts import Abort, For, Sequential
from tilefoundry.ir.types import DType, TensorType, UnitType
from tilefoundry.parser.ast_pattern import ParseError
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


@prim_func(target=CpuTarget())
def current_tir_ops(
    a: Tensor[(8,), "f32"],
    b: Tensor[(8,), "f32"],
    weight: Tensor[(8,), "f32"],
):
    ptr = T.ptr_of(a)
    span = T.memory_span(ptr)
    T.binary(span, b, b, kind=BinaryKind.ADD)
    T.unary(b, b, kind=UnaryKind.NEG)
    T.clamp(b, b, min_val=-1.0, max_val=1.0)
    T.reduce(b, b, b, axes=(0,), kind=ReduceKind.SUM)
    T.relu(b, b)
    T.rms_norm(b, b, weight, eps=1e-5)
    return


CURRENT_TIR_SURFACE_PROGRAMS = {
    "mma": (MmHandwritten, "MmHandwritten"),
    "sync": (SyncSquare, "SyncSquare"),
    "async_copy": (AsyncStage, "AsyncStage"),
    "remaining_ops": (current_tir_ops, "current_tir_ops"),
}


def _assert_lint_clean(source: str) -> None:
    lint = subprocess.run(
        [
            "ruff",
            "check",
            "--config",
            "pyproject.toml",
            "--stdin-filename",
            "tests/fixtures/tir_printed.py",
            "-",
        ],
        input=source,
        text=True,
        capture_output=True,
        check=False,
    )
    assert lint.returncode == 0, lint.stdout + lint.stderr


def test_tir_module_printer_roundtrips() -> None:
    printed = as_script(_TirBaseline)
    assert as_script(import_dsl(printed, name="_TirBaseline")) == printed


@pytest.mark.parametrize(
    ("program", "binding"),
    [pytest.param(*program, id=name) for name, program in CURRENT_TIR_SURFACE_PROGRAMS.items()],
)
def test_current_tir_surface_roundtrips(program, binding: str) -> None:
    printed = as_script(program)
    assert as_script(import_dsl(printed, name=binding)) == printed
    _assert_lint_clean(printed)


def test_current_tir_surface_programs_cover_every_schema_form() -> None:
    lines = [
        line.strip()
        for program, _ in CURRENT_TIR_SURFACE_PROGRAMS.values()
        for line in as_script(program).splitlines()
    ]
    for name, form in TIR_SUPPORT_MATRIX.items():
        if form == "effect-form":
            assert any(line.startswith(f"T.{name}(") for line in lines), name
        else:
            assert any(f" = T.{name}(" in line for line in lines), name


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


def test_tir_for_if_and_abort_roundtrip() -> None:
    function = import_dsl(
        "from tilefoundry import prim_func\n"
        "from tilefoundry.dsl import Tensor\n"
        "from tilefoundry.target import CpuTarget\n\n"
        "@prim_func(target=CpuTarget())\n"
        "def control(a: Tensor[(1,), 'f32']):\n"
        "    for i in range(2):\n"
        "        if i < 1:\n"
        "            abort('stop')\n",
        name="control",
    )
    printed = as_script(function)
    assert "for i in range(0, 2, 1):" in printed
    assert "if i < 1:" in printed
    assert as_script(import_dsl(printed, name="control")) == printed


def test_tir_for_if_and_sync_mesh_forms_roundtrip() -> None:
    function = import_dsl(
        "from tilefoundry import prim_func\n"
        "from tilefoundry.dsl import T, Tensor\n"
        "from tilefoundry.ir.types.shard import Layout, Mesh, Topology\n"
        "from tilefoundry.target import CudaTarget\n\n"
        "@prim_func(target=CudaTarget('nvidia.h200_sxm'))\n"
        "def device(a: Tensor[(64,), 'f32'], out: Tensor[(64,), 'f32']):\n"
        "    with Mesh((Topology('thread', 32),), Layout((32,), (1,))) as thread:\n"
        "        for i in range(0, 2, 1):\n"
        "            if i < 1:\n"
        "                T.sync(thread)\n"
        "            else:\n"
        "                T.sync(thread[:])\n",
        name="device",
    )
    printed = as_script(function)
    assert "T.sync(thread)" in printed
    assert "T.sync(Mesh(" in printed
    assert as_script(import_dsl(printed, name="device")) == printed


def test_tir_for_rejects_nonconstant_bounds() -> None:
    with pytest.raises(ParseError, match="CUDA codegen cannot emit a non-constant loop bound"):
        import_dsl(
            "from tilefoundry import prim_func\n"
            "from tilefoundry.dsl import Tensor\n"
            "from tilefoundry.target import CpuTarget\n\n"
            "@prim_func(target=CpuTarget())\n"
            "def dynamic(n: Tensor[(), 'i64']):\n"
            "    for i in range(n):\n"
            "        return\n",
            name="dynamic",
        )


def test_tir_for_codegen_rejects_nonconstant_bounds() -> None:
    scalar = TensorType.scalar(DType.i64)
    variable = Var(type=scalar, name="n")
    loop = For(
        induction_var=Var(type=scalar, name="i"),
        start=Constant(type=scalar, value=0),
        stop=variable,
        step=Constant(type=scalar, value=1),
        body=Sequential(()),
    )
    with pytest.raises(NotImplementedError, match="silently wrong loop"):
        CodegenContext().emit_node(loop)


def test_tir_shape_of_roundtrips() -> None:
    function = import_dsl(
        "from tilefoundry import prim_func\n"
        "from tilefoundry.dsl import Tensor\n"
        "from tilefoundry.target import CpuTarget\n\n"
        "@prim_func(target=CpuTarget())\n"
        "def shape(a: Tensor[(8,), 'f32']):\n"
        "    extent = shape_of(a, axis=0)\n"
        "    return\n",
        name="shape",
    )
    printed = as_script(function)
    assert as_script(import_dsl(printed, name="shape")) == printed


def test_tir_dispatch_call_roundtrips_with_statement_fallback() -> None:
    module = Module(
        name="Dispatch",
        functions=tuple(build_dispatch_functions()),
        entry="main",
    )
    printed = as_script(module)
    assert "with dispatch_call(" in printed
    assert "\n            abort('')" in printed
    assert as_script(import_dsl(printed, name="Dispatch")) == printed
    _assert_lint_clean(printed)


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
