"""Canonical fixture contracts for the authored HIR and TIR surfaces."""

from __future__ import annotations

from tests.core._source import import_dsl
from tilefoundry.dsl import *
from tilefoundry.inspection import as_script


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
