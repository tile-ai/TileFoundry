"""The scheduling corpus's plain programs, read the way a scheduler reads them."""

from __future__ import annotations

from functools import partial
from pathlib import Path

import pytest

from tests.core.schedule._fixtures import FixtureCase, arithmetic_facts, check_facts, check_fixture

_AMX_FACTS_TEXT = f"""\
target apple.m2_pro
instructions
  T.amx_fma16   amx_outer_product
  T.amx_load_x  amx_outer_product
  T.amx_load_y  amx_outer_product
  T.amx_store   amx_outer_product
{arithmetic_facts(12)}  T.copy        all targets
  T.relu        all targets
  T.reduce      all targets
"""


FIXTURES = (
    FixtureCase(
        "Apple facts list AMX and shared instructions",
        ("schedule", "facts", "--target", "apple.m2_pro", "{out}"),
        partial(check_facts, expected=_AMX_FACTS_TEXT),
    ),
)


@pytest.mark.parametrize("case", FIXTURES, ids=lambda case: case.name)
def test_fixture(
    case: FixtureCase, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    check_fixture(case, tmp_path, capsys)
