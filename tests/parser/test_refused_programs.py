"""The one entry point for every subject ``tests/parser`` refuses.

The table is in ``error_cases.py``; this file only runs it. A refusal that is not
a row here is a refusal nothing pins.
"""

from __future__ import annotations

import pytest

from tests.parser.error_cases import ERROR_CASES, ParseErrorCase, run_parse_error_case


@pytest.mark.parametrize("case", ERROR_CASES, ids=[case.id for case in ERROR_CASES])
def test_a_refused_program_raises_its_diagnostic(case: ParseErrorCase) -> None:
    """Every refused subject named in the table is refused, saying why."""
    run_parse_error_case(case)
