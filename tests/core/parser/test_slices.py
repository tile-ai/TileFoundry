"""Parser diagnostics for authored tile-window slices."""

from __future__ import annotations

from pathlib import Path

import pytest

from tilefoundry.cli.source import load_authored_ir
from tilefoundry.parser import ParseError

DIAGNOSTICS = Path(__file__).parents[1] / "fixtures" / "diagnostics"


def test_tile_window_bound_rejected() -> None:
    with pytest.raises(
        ParseError,
        match=r"tile loop variable is already a window.*x\[:, t, :\].*base = t \+ 0",
    ):
        load_authored_ir(str(DIAGNOSTICS / "parser_tile_window_bound.py"))


def test_tile_window_rejects_division() -> None:
    with pytest.raises(ParseError, match=r"tile windows only support ± c and \* c"):
        load_authored_ir(str(DIAGNOSTICS / "parser_tile_window_division.py"))


def test_literal_index_rejects_division() -> None:
    with pytest.raises(ParseError, match=r"index arithmetic uses //"):
        load_authored_ir(str(DIAGNOSTICS / "parser_literal_index_division.py"))


def test_symbolic_index_rejects_division() -> None:
    with pytest.raises(ParseError, match=r"index arithmetic uses //"):
        load_authored_ir(str(DIAGNOSTICS / "parser_symbolic_index_division.py"))
