"""Parser ownership of function-only docstring semantics."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests.core._corpus import reload
from tilefoundry import cli, inspection, parser
from tilefoundry.dsl import Mesh, func, pattern
from tilefoundry.ir import core

FIXTURES = Path(__file__).parents[1] / "fixtures"
LOOP_ITERATORS = {
    "tf.tile(10)": "parser_tile_one_argument.py",
    "tf.tile(1, 2, 3, 4)": "parser_tile_four_arguments.py",
    "range(1, 2, 3, 4)": "parser_range_four_arguments.py",
    "steps(1, 2)": "parser_unknown_loop_iterator.py",
}


def test_function_accepts_leading_docstring() -> None:
    documented = cli.load_authored_ir(
        f"{FIXTURES}/parser/documented_function.py:Documented"
    ).entry_function()

    assert "This is documentation" not in inspection.as_script(documented)


@pytest.mark.parametrize(
    ("iterator", "expected"),
    (
        ("tf.tile(10)", "tf.tile(extent) is not supported; use range(extent)"),
        (
            "tf.tile(1, 2, 3, 4)",
            "tf.tile() takes 2 or 3 arguments, (stop, step) or (start, stop, step), got 4",
        ),
        ("range(1, 2, 3, 4)", "range() takes 1 to 3 arguments, got 4"),
        ("steps(1, 2)", "loop iterator must be tf.tile(...) or range(...)"),
    ),
)
def test_loop_iterator_names_arity_error(iterator: str, expected: str) -> None:
    """The shape-exact syntax must not reduce these to a bare match failure.

    Encoding each iterator's arity is what lets the generated grammar show the
    accepted forms, so the reason is stated before the shape rejects.
    """
    with pytest.raises(parser.ParseError, match=re.escape(expected)):
        cli.load_authored_ir(str(FIXTURES / "diagnostics" / LOOP_ITERATORS[iterator]))


def test_lowercase_tuple_annotation_is_refused() -> None:
    """Only ``Tuple[...]`` names a tuple type; the lowercase spelling is not DSL."""
    with pytest.raises(parser.ParseError, match=":12:24"):
        cli.load_authored_ir(str(FIXTURES / "diagnostics" / "parser_lowercase_tuple_annotation.py"))


@pytest.mark.parametrize("module", ("Pipeline", "TwoMeshes"))
def test_class_body_meshes_print_in_the_class_and_reparse(module) -> None:
    """A mesh the class body declares is named by signatures and bodies, and prints back there."""
    loaded = cli.load_authored_ir(f"{FIXTURES / 'parser' / 'class_meshes.py'}:{module}")
    printed = inspection.as_script(loaded)
    declared = [line.strip() for line in printed.splitlines() if " = Mesh(" in line]
    assert declared and all(line.startswith(("gpu = ", "gpu_2 = ")) for line in declared)
    assert "@func(mesh=" not in printed
    again = reload(printed, loaded.name)
    assert inspection.as_script(again) == printed
    for before, after in zip(loaded.functions, again.functions, strict=True):
        assert [param.type for param in before.params] == [param.type for param in after.params]


def test_func_and_specialize_take_no_mesh() -> None:
    """A mesh is declared in the class body; the decorators have no mesh argument."""
    mesh = Mesh(("gpu",), (2,), names=("g",))
    with pytest.raises(TypeError, match="mesh"):
        func(mesh=mesh)
    with pytest.raises(TypeError, match="mesh"):
        core.Function.specialize(None, pattern.Range("n", 0, 4), mesh=mesh)
