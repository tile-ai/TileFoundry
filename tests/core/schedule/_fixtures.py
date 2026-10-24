"""Schedule CLI fixture cases, execution, and facts report checks."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import torch

from tilefoundry.cli import main as cli_main


@dataclass(frozen=True)
class CliCase:
    name: str
    argv: tuple[str, ...]
    check: Callable[[Path, pytest.CaptureResult[str]], None]
    code: int = 0
    files: Mapping[str, Callable[[], torch.Tensor]] = field(default_factory=dict)


def _quiet_json(out: Path, captured: pytest.CaptureResult[str]) -> dict:
    assert captured == ("", "")
    return json.loads(out.read_text())


def check_facts(
    out: Path,
    captured: pytest.CaptureResult[str],
    *,
    expected: str | dict,
) -> None:
    if isinstance(expected, dict):
        assert _quiet_json(out, captured) == expected
    else:
        assert captured == ("", "")
        assert out.read_text() == expected


def check_cli_case(case: CliCase, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    for name, make in case.files.items():
        torch.save(make(), tmp_path / name)
    out = tmp_path / "out"
    argv = [arg.format(out=out, tmp=tmp_path) for arg in case.argv]
    try:
        code = cli_main(argv)
    except SystemExit as error:
        code = error.code
    assert code == case.code
    case.check(out, capsys.readouterr())


def cli_case_test(cases):
    @pytest.mark.parametrize("case", cases, ids=lambda case: case.name)
    def test_cli_case(case, tmp_path, capsys):
        check_cli_case(case, tmp_path, capsys)

    return test_cli_case
