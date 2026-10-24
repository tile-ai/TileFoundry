"""Tests and helpers for the owning target."""

from __future__ import annotations

from functools import partial

import pytest

from tests.core.schedule._fixtures import FixtureCase, _check_facts, check_fixture

_AMX_FACTS_TEXT = """\
target apple.m2_pro
instructions
  T.amx_fma16   amx_outer_product
  T.amx_load_x  amx_outer_product
  T.amx_load_y  amx_outer_product
  T.amx_store   amx_outer_product
  T.arange      all targets
  T.binary      all targets
  T.cast        all targets
  T.clamp       all targets
  T.unary       all targets
  T.where       all targets
  T.copy        all targets
  T.relu        all targets
  T.reduce      all targets
"""


FIXTURES = (FixtureCase("Apple facts list AMX and shared instructions", ("schedule", "facts", "--target", "apple.m2_pro", "{out}"), partial(_check_facts, expected=_AMX_FACTS_TEXT)),)

@pytest.mark.parametrize("case", FIXTURES, ids=lambda case: case.name)
def test_fixture(case, tmp_path, capsys):
    check_fixture(case, tmp_path, capsys)
