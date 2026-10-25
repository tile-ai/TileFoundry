"""A translation unit is assembled from its functions, in one place."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.core._corpus import Case, case, check_case, special
from tilefoundry.codegen.context import EmitContext
from tilefoundry.codegen.context_builder import build_codegen_context
from tilefoundry.codegen.linkable_function import LinkableFunction
from tilefoundry.codegen.linkable_module import LinkableModule

SQUARE = Path(__file__).parents[1] / "fixtures" / "tir" / "square.py"


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


def _host_unit(square) -> LinkableModule:
    """The host translation unit of a module whose entry launches a shape-dispatched kernel."""
    entry = square.entry_function()
    root = build_codegen_context(square)
    group = next(group for group in root.groups if entry in group.functions)
    view = root.for_group(group)
    return entry.target.resolve(
        LinkableModule, square, (entry,), entry.target.resolve(EmitContext, view)
    )


def test_unit_orders_declarations_before_definitions() -> None:
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


class LinkableCase(Case):
    """The host translation unit emitted for one program."""


@case
class SquareHostUnit(LinkableCase):
    path: Path = SQUARE

    def guard(self, state, problems):
        unit = _host_unit(state)
        problems.run("declaration matches definition", self.declaration_matches_definition, unit)
        problems.run(
            "entry declared before defined", self.entry_declared_before_defined, unit, state
        )

    @staticmethod
    def declaration_matches_definition(unit):
        entry = unit.functions[0]
        assert _param_types(entry.declaration) == _param_types(entry.definition.split("{")[0])

    @staticmethod
    def entry_declared_before_defined(unit, program):
        entry = unit.functions[0]
        assert entry.name == program.entry
        assert unit.source.index(entry.declaration) < unit.source.index(entry.definition)
        assert entry.declaration not in unit.preamble


CASES = special(LinkableCase)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_linkable_case(case: LinkableCase) -> None:
    check_case(case)
