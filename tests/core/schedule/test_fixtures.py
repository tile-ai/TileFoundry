"""CPU scheduling facts for target-neutral instructions."""

from __future__ import annotations

from functools import partial
from pathlib import Path

import pytest

from tests.core.schedule._fixtures import CliCase, check_cli_case, check_facts

_TARGET_NEUTRAL_FACTS = """\
target {target}
instructions
  T.arange  all targets
  T.binary  all targets
  T.cast    all targets
  T.clamp   all targets
  T.unary   all targets
  T.where   all targets
  T.copy    all targets
  T.relu    all targets
  T.reduce  all targets
"""


FIXTURES = (
    CliCase(
        "CPU facts list only target neutral instructions",
        ("schedule", "facts", "--target", "cpu", "{out}"),
        partial(check_facts, expected=_TARGET_NEUTRAL_FACTS.format(target="cpu")),
    ),
)


@pytest.mark.parametrize("case", FIXTURES, ids=lambda case: case.name)
def test_cli_case(
    case: CliCase, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    check_cli_case(case, tmp_path, capsys)
