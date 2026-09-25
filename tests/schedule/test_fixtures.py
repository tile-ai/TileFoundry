"""The scheduling corpus's plain programs, read the way a scheduler reads them.

These are the programs an author states before choosing an instruction: a tiled
matmul chain, the same chain with both operand windows staged, its untiled
baseline, and the real matmul over a CTA grid. What each one is here to witness
is that it parses, holds together, and can be measured -- the three questions
anything downstream asks before it offers a schedule at all.
"""

from __future__ import annotations

import importlib

import pytest

from tilefoundry.analysis.api import analyze
from tilefoundry.analysis.check import check_program

PLAIN = (
    "gemm_8192x17408x5120_cta_grid",
    "gemm_relu_gemm_smem_staged",
    "gemm_relu_gemm_tiled",
    "gemm_relu_gemm_untiled",
)


@pytest.mark.parametrize("name", PLAIN)
def test_plain_program_is_analyzable(name: str) -> None:
    module = importlib.import_module(f"tests.fixtures.schedule.plain.{name}")
    program = next(
        value for value in vars(module).values() if type(value).__name__ == "Module"
    )
    entry = next(function for function in program.functions if function.name == "gemm")
    check_program(program, entry)
    result = analyze(program, entry, analysis=("memory", "performance"))
    assert result.metadata_types
