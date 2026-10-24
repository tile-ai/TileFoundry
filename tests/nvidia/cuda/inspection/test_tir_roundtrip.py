"""Canonical fixture contracts for the authored HIR and TIR surfaces."""

from __future__ import annotations

from pathlib import Path

import pytest

import tilefoundry_nvidia.cuda.codegen  # noqa: F401
from tests.core._source import import_dsl
from tests.core.inspection._tir_roundtrip import _module_in, check_tir_roundtrip
from tilefoundry.dsl import *
from tilefoundry.inspection import as_script
from tilefoundry.ir.core import Constant, Var
from tilefoundry.ir.tir.stmts import For, Sequential
from tilefoundry.visitor_registry.verify import verify_module
from tilefoundry_nvidia.cuda.codegen.context import CudaCodegenContext

FIXTURES = Path(__file__).parents[1] / "fixtures"
CANONICAL = tuple(path for path in (FIXTURES / "tir").glob("*.py") if path.name != "layouts.py")
SUGAR = FIXTURES / "inspection" / "type_printer_sugar.py"


def test_placed_types_print_and_reparse_as_layout_sugar() -> None:
    """Placement sugar is the whole type surface, and it parses back to itself.

    The golden is the review surface: a value the sugar cannot state would
    appear here as the verbose ``ShardLayout(...)`` form instead, and a mesh
    axis the sugar named wrongly would not survive the reparse.
    """
    printed = as_script(_module_in(SUGAR))

    assert printed == SUGAR.with_suffix(".printed.txt").read_text()
    assert as_script(import_dsl(printed, name="TypePrinterSugar")) == printed


def test_tir_for_if_and_sync_mesh_forms_roundtrip() -> None:
    function = import_dsl(
        (
            "from tilefoundry.dsl import *\n"
            "\n"
            "\n"
            "from tilefoundry_nvidia import CudaTarget\n"
            "\n"
            "@prim_func(target=CudaTarget('nvidia.h200_sxm'))\n"
            "def device(a: Tensor[(64,), 'f32'], out: Tensor[(64,), 'f32']):\n"
            "    with Mesh((Topology('thread', 32),), Layout((32,), (1,))) as thread:\n"
            "        for i in range(0, 2, 1):\n"
            "            if i < 1:\n"
            "                T.sync(thread)\n"
            "            else:\n"
            "                T.sync(thread[:])\n"
        ),
        name="device",
    )
    printed = as_script(function)
    assert "T.sync(thread)" in printed
    assert "T.sync(thread[:])" in printed
    assert as_script(import_dsl(printed, name="device")) == printed


def test_launch_options_roundtrip() -> None:
    module = import_dsl(
        (
            "from tilefoundry.dsl import *\n"
            "from tilefoundry.target import CpuTarget\n"
            "from tilefoundry_nvidia import CudaTarget\n"
            "\n"
            "@module(entry='host', topologies=(Topology('thread', 32),))\n"
            "class Launched:\n"
            "    @prim_func(target=CudaTarget('nvidia.h200_sxm'))\n"
            "    def device(a: Tensor[(32,), 'f32']):\n"
            "        pass\n"
            "\n"
            "    @prim_func(target=CpuTarget())\n"
            "    def host(a: Tensor[(32,), 'f32']):\n"
            "        launch(device, a, grid=(1, 1, 1), block=(32, 1, 1), cluster=(1, 1, 1),\n"
            "               dynamic_smem=512)\n"
        ),
        name="Launched",
    )
    printed = as_script(module)
    assert "cluster=(1, 1, 1), dynamic_smem=512)" in printed
    assert as_script(import_dsl(printed, name="Launched")) == printed


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
    context = CudaCodegenContext()
    context.emit_node(loop)
    assert "i_1 < n_2" in context.source()


def test_long_statement_sequence_prints() -> None:
    statements = "\n".join(
        f'            value_{i} = T.alloc_tensor(tensor_type=Tensor[(), "f32", "rmem"])'
        for i in range(1000)
    )
    module = import_dsl(
        'from tilefoundry.dsl import *\n'
        'from tilefoundry_nvidia import CudaTarget\n'
        '@module()\n'
        'class LongSequence:\n'
        '    @prim_func(target=CudaTarget("nvidia.h200_sxm"))\n'
        '    def sequence():\n'
        '        with Mesh((Topology("thread", 32),), layout=(32,)) as thread:\n'
        f'{statements}\n',
    )
    printed = as_script(module)
    assert printed.count("T.alloc_tensor(") == 1000
    verify_module(module)
    context = CudaCodegenContext(target=module.functions[0].target)
    context.emit_node(module.functions[0].body)
    assert context.source()


@pytest.mark.parametrize(
    "path",
    sorted(path for path in CANONICAL if path.name != "__init__.py")
    + sorted((FIXTURES / "scaffold").glob("tir_*.py")),
    ids=lambda path: path.stem,
)
def test_tir_roundtrip(path: Path) -> None:
    check_tir_roundtrip(path, CANONICAL)
