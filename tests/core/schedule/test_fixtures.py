"""The scheduling corpus's plain programs, read the way a scheduler reads them.

These are the programs an author states before choosing an instruction: a tiled
matmul chain, the same chain with both operand windows staged, its untiled
baseline, and the real matmul over a CTA grid. What each one is here to witness
is that it parses, holds together, and can be measured -- the three questions
anything downstream asks before it offers a schedule at all.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

import pytest

from tests.core.schedule._fixtures import FixtureCase, check_facts, check_fixture

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
