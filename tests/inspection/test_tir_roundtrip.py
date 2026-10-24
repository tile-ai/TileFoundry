"""Canonical fixture contracts for the authored HIR and TIR surfaces."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

import tilefoundry_nvidia.cuda.codegen  # noqa: F401
from tests._source import import_dsl
from tilefoundry.dsl import *
from tilefoundry.inspection import as_script
from tilefoundry.ir.core import Constant, Var
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.module_utils import list_call_paths
from tilefoundry.ir.tir.stmts import For, Sequential
from tilefoundry.visitor_registry.verify import verify_module
from tilefoundry_nvidia.cuda.codegen.context import CudaCodegenContext


def _module_functions(module):
    return tuple(frame.function for frame in list_call_paths(module).values())


FIXTURES = Path(__file__).parents[1] / "fixtures"
CANONICAL = tuple(path for path in (FIXTURES / "tir").glob("*.py") if path.name != "layouts.py")
SUGAR = FIXTURES / "inspection" / "type_printer_sugar.py"


def _module_in(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    modules = [value for value in vars(loaded).values() if isinstance(value, Module)]
    return loaded.Root if path.parent.name == "scaffold" else modules[0]


@pytest.mark.parametrize(
    "path",
    sorted(path for path in CANONICAL if path.name != "__init__.py")
    + sorted((FIXTURES / "scaffold").glob("tir_*.py")),
    ids=lambda path: path.stem,
)
def test_tir_roundtrip(path: Path) -> None:
    module = _module_in(path)
    printed = as_script(module)
    if path in CANONICAL:
        assert printed == path.read_text()
    rebuilt = import_dsl(printed, name=module.name)
    assert as_script(rebuilt) == printed
    verify_module(rebuilt)
    original = [param.is_const for fn in _module_functions(module) for param in fn.params]
    restored = [param.is_const for fn in _module_functions(rebuilt) for param in fn.params]
    assert restored == original
    if any(original):
        assert "ConstTensor[" in printed
    if path.stem == "tir_launches":
        assert "launch(child.run," in printed


def test_placed_types_print_and_reparse_as_layout_sugar() -> None:
    """Placement sugar is the whole type surface, and it parses back to itself.

    The golden is the review surface: a value the sugar cannot state would
    appear here as the verbose ``ShardLayout(...)`` form instead, and a mesh
    axis the sugar named wrongly would not survive the reparse.
    """
    printed = as_script(_module_in(SUGAR))

    assert printed == SUGAR.with_suffix(".printed.txt").read_text()
    assert as_script(import_dsl(printed, name="TypePrinterSugar")) == printed


def test_mixed_hir_tir_module_prints_both_function_families() -> None:
    mixed = import_dsl(
        (
            "from tilefoundry.dsl import *\n"
            "\n"
            "from tilefoundry.target import CpuTarget\n"
            "\n"
            "@module()\n"
            "class Mixed:\n"
            "    @func\n"
            "    def h(a: Tensor[(1,), 'f32']):\n"
            "        return a\n"
            "\n"
            "    @prim_func(target=CpuTarget())\n"
            "    def t(a: Tensor[(1,), 'f32']):\n"
            "        return\n"
        ),
        name="Mixed",
    )
    printed = as_script(mixed)
    assert "@func" in printed and "@prim_func" in printed
    assert as_script(import_dsl(printed, name="Mixed")) == printed


def test_tir_for_if_and_abort_roundtrip() -> None:
    function = import_dsl(
        (
            "from tilefoundry.dsl import *\n"
            "\n"
            "from tilefoundry.target import CpuTarget\n"
            "\n"
            "@prim_func(target=CpuTarget())\n"
            "def control(a: Tensor[(1,), 'f32']):\n"
            "    with Mesh((Topology('thread', 32),), Layout((32,), (1,))) as thread:\n"
            "        for i in range(2):\n"
            "            if i < 1:\n"
            "                T.abort(message='stop')\n"
        ),
        name="control",
    )
    printed = as_script(function)
    assert "for i in range(0, 2, 1):" in printed
    assert "if i < 1:" in printed
    assert as_script(import_dsl(printed, name="control")) == printed


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


def test_tir_for_accepts_nonconstant_bounds() -> None:
    import_dsl(
        (
            "from tilefoundry.dsl import *\n"
            "\n"
            "from tilefoundry.target import CpuTarget\n"
            "\n"
            "@prim_func(target=CpuTarget())\n"
            "def dynamic(n: Tensor[(), 'i64']):\n"
            "    for i in range(n):\n"
            "        return\n"
        ),
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
