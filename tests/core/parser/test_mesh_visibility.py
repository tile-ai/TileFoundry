"""Placed values are consumed only where their positions and storage reach."""

from __future__ import annotations

from pathlib import Path

import pytest

from tilefoundry.cli.source import load_authored_ir
from tilefoundry.ir.core import VerifyError
from tilefoundry.parser import ParseError

_DIAGNOSTICS = Path(__file__).parents[1] / "fixtures" / "diagnostics"
_PARSER = Path(__file__).parents[1] / "fixtures" / "parser"


@pytest.mark.parametrize(
    "program",
    (
        "add_before_mesh",
        "add_after_mesh",
        "call_before_mesh",
        "call_after_mesh",
        "binary_before_mesh",
        "binary_after_mesh",
        "unary_before_mesh",
        "unary_after_mesh",
        "matmul_before_mesh",
        "matmul_after_mesh",
        "slice_before_mesh",
        "slice_after_mesh",
        "tuple_before_mesh",
        "tuple_after_mesh",
        "scalar_before_mesh",
        "scalar_after_mesh",
    ),
)
def test_runtime_expressions_require_mesh_scope(program) -> None:
    with pytest.raises(ParseError, match="runs outside every mesh scope"):
        load_authored_ir(str(_DIAGNOSTICS / f"parser_unscoped_{program}.py"))


def test_return_after_mesh_requires_scope() -> None:
    with pytest.raises(ParseError, match="runs outside every mesh scope"):
        load_authored_ir(str(_DIAGNOSTICS / "parser_unscoped_return.py"))


def test_logical_tir_calls_stay_unscoped() -> None:
    logical = load_authored_ir(f"{_PARSER}/unscoped_calls.py:Logical").entry_function()
    tir = load_authored_ir(f"{_PARSER}/unscoped_calls.py:Tir").entry_function()

    assert logical.body is not None and tir.body is not None


def test_finer_layout_than_scope_rejected() -> None:
    with pytest.raises(
        VerifyError,
        match=(
            "input 0 is laid out more finely than the scope it runs in; "
            "write it inside that scope, or reshard it back first"
        ),
    ):
        load_authored_ir(str(_DIAGNOSTICS / "value_escapes.py"))


def test_coarse_rmem_layout_rejected() -> None:
    with pytest.raises(
        VerifyError,
        match=(
            "input 0 is laid out more coarsely and kept in rmem, which does not "
            "reach the units this runs on; reshard it to smem or gmem first"
        ),
    ):
        load_authored_ir(str(_DIAGNOSTICS / "coarse_rmem.py"))


def test_reshard_moves_invisible_value() -> None:
    ReshardEscape = load_authored_ir(str(_PARSER / "reshard_escape.py"))

    assert ReshardEscape.entry_function().body is not None
