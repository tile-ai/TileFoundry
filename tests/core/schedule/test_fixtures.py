"""The scheduling corpus's plain programs, read the way a scheduler reads them."""

from __future__ import annotations

from functools import partial
from pathlib import Path

import pytest

from tests.core.schedule._fixtures import FixtureCase, arithmetic_facts, check_facts, check_fixture

_TARGET_NEUTRAL_FACTS = f"""\
target {{target}}
instructions
{arithmetic_facts(8)}  T.copy    all targets
  T.relu    all targets
  T.reduce  all targets
"""


FIXTURES = (
    FixtureCase(
        "CPU facts list only target neutral instructions",
        ("schedule", "facts", "--target", "cpu", "{out}"),
        partial(check_facts, expected=_TARGET_NEUTRAL_FACTS.format(target="cpu")),
    ),
)


@pytest.mark.parametrize("case", FIXTURES, ids=lambda case: case.name)
def test_fixture(
    case: FixtureCase, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    check_fixture(case, tmp_path, capsys)
