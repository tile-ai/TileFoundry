"""A translation unit is assembled from its functions, in one place."""

from __future__ import annotations

from tests.fixtures.tir.square import TirSquare
from tilefoundry.codegen.cpu.context import CpuCodegenContext
from tilefoundry.codegen.cpu.module import emit_host_module
from tilefoundry.codegen.linkable import LinkableFunction, LinkableModule
from tilefoundry.codegen.signature import symbol_table
from tilefoundry.target import CudaTarget


def _linkable_function(name: str) -> LinkableFunction:
    return LinkableFunction(
        name=name,
        declaration=f"void {name}(int);",
        definition=f"void {name}(int x) {{ {name}_body(x); }}",
    )


def _param_types(parameter_list: str) -> list[str]:
    """The C++ types of a parameter list, with any parameter names dropped."""
    inner = parameter_list[parameter_list.index("(") + 1 : parameter_list.rindex(")")]
    return [part.strip().rsplit(" ", 1)[0].strip() for part in inner.split(",")]


def _host_unit() -> LinkableModule:
    """The host translation unit of a module whose entry launches a shape-dispatched kernel."""
    entry = TirSquare.entry_function()
    ctx = CpuCodegenContext(
        symbols=symbol_table(TirSquare, CudaTarget("nvidia.h200_sxm")), target=entry.target
    )
    return emit_host_module(TirSquare, (entry,), entry.target, ctx)


def test_a_unit_is_its_preamble_then_every_declaration_then_every_definition() -> None:
    unit = LinkableModule(
        target="cuda",
        language="cu",
        preamble="#include <one.h>",
        functions=(_linkable_function("first"), _linkable_function("second")),
    )
    written = [
        unit.source.index(text)
        for text in (
            "#include <one.h>",
            "void first(int);",
            "void second(int);",
            "first_body",
            "second_body",
        )
    ]
    assert written == sorted(written)


def test_a_unit_with_no_function_is_its_preamble() -> None:
    unit = LinkableModule(target="cpu", language="cpp", preamble="#include <one.h>")
    assert unit.source == "#include <one.h>\n"


def test_a_declaration_and_its_definition_cannot_disagree_on_the_parameters() -> None:
    entry = _host_unit().functions[0]
    assert _param_types(entry.declaration) == _param_types(entry.definition.split("{")[0])


def test_the_host_entry_is_declared_before_the_unit_defines_it() -> None:
    unit = _host_unit()
    entry = unit.functions[0]
    assert entry.name == TirSquare.entry
    assert unit.source.index(entry.declaration) < unit.source.index(entry.definition)
    assert entry.declaration not in unit.preamble
