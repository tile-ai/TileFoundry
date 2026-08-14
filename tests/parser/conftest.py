"""The golden-file fixture for the parser programs.

A golden is the program printed back as DSL source, so what a reviewer reads is
the program itself rather than a list of node assertions. ``--update-golden``
rewrites the files from what the parser produced instead of asserting against
them; run it whenever the printer changes, and read the diff before keeping it.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from pathlib import Path

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register ``--update-golden``."""
    parser.addoption(
        "--update-golden",
        action="store_true",
        help="rewrite tests/parser/golden/*.py from what the parser produced",
    )


@dataclass(frozen=True)
class GoldenFiles:
    """The recorded output of each program, under one directory."""

    root: Path
    update: bool

    def check(self, name: str, actual: str) -> None:
        """Compare *actual* against the recorded ``name``, or record it."""
        path = self.root / name
        if self.update:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(actual)
            return
        assert path.exists(), f"no golden at {path}; rerun with --update-golden"
        expected = path.read_text()
        if actual == expected:
            return
        diff = "".join(
            difflib.unified_diff(
                expected.splitlines(keepends=True),
                actual.splitlines(keepends=True),
                fromfile=f"{name} (recorded)",
                tofile=f"{name} (parsed now)",
            )
        )
        raise AssertionError(
            f"{name} no longer matches what the parser produces:\n\n{diff}\n"
            "If the new output is right, rerun with --update-golden."
        )


@pytest.fixture
def golden(request: pytest.FixtureRequest) -> GoldenFiles:
    """The golden directory beside this file, honouring ``--update-golden``."""
    return GoldenFiles(
        root=Path(__file__).parent / "golden",
        update=bool(request.config.getoption("--update-golden")),
    )
