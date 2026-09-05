"""Canonical fixture contracts for the authored HIR and TIR surfaces."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

import tilefoundry.codegen.cuda  # noqa: F401
from tests._source import import_dsl
from tests.ir.test_dispatch_call import _build_module as build_dispatch_functions
from tilefoundry.codegen.cuda.context import CodegenContext
from tilefoundry.inspection import as_script
from tilefoundry.ir.core import Constant, Var
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.tir.stmts import For, Sequential
from tilefoundry.ir.types import DType, TensorType

FIXTURES = Path(__file__).parents[1] / "fixtures"
CANONICAL = (
    *(FIXTURES / "tir").glob("*.py"),
    *(FIXTURES / "logical").glob("canonical_*.py"),
)


def _module_in(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return next(value for value in vars(loaded).values() if isinstance(value, Module))


@pytest.mark.parametrize(
    "path",
    sorted(path for path in CANONICAL if path.name != "__init__.py"),
    ids=lambda path: path.stem,
)
def test_fixture_prints_back_to_its_own_source(path: Path) -> None:
    assert as_script(_module_in(path)) == path.read_text()


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
        "from tilefoundry.dsl import T, Tensor\n"
        "from tilefoundry.target import CpuTarget\n\n"
        "@prim_func(target=CpuTarget())\n"
        "def control(a: Tensor[(1,), 'f32']):\n"
        "    for i in range(2):\n"
        "        if i < 1:\n"
        "            T.abort(message='stop')\n",
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


def test_tir_for_accepts_nonconstant_bounds() -> None:
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


def test_tir_for_codegen_renders_nonconstant_bounds() -> None:
    scalar = TensorType.scalar(DType.i64)
    variable = Var(type=scalar, name="n")
    loop = For(
        induction_var=Var(type=scalar, name="i"),
        start=Constant(type=scalar, value=0),
        stop=variable,
        step=Constant(type=scalar, value=1),
        body=Sequential(()),
    )
    context = CodegenContext()
    context.emit_node(loop)
    assert "i_1 < n_2" in context.source()


def test_lowered_dispatch_prints_readable_shape_and_fallback() -> None:
    module = Module(
        name="Dispatch",
        functions=tuple(build_dispatch_functions()),
        entry="main",
    )

    printed = as_script(module)

    assert "with dispatch_call(" in printed
    assert "shape_of(" in printed
    assert "T.abort(message='')" in printed


    return None
