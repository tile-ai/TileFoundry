"""The scheduling corpus's plain programs, read the way a scheduler reads them.

These are the programs an author states before choosing an instruction: a tiled
matmul chain, the same chain with both operand windows staged, its untiled
baseline, and the real matmul over a CTA grid. What each one is here to witness
is that it parses, holds together, and can be measured -- the three questions
anything downstream asks before it offers a schedule at all.
"""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path

import pytest

from tilefoundry.analysis.api import analyze
from tilefoundry.analysis.check import check_program
from tilefoundry.inspection import as_script
from tilefoundry.ir.tir import PrimFunction
from tilefoundry.ir.tir.verify import verify_prim_function

PLAIN = (
    "gemm_8192x17408x5120_cta_grid",
    "gemm_relu_gemm_smem_staged",
    "gemm_relu_gemm_tiled",
    "gemm_relu_gemm_untiled",
)
TIR = tuple(sorted((Path(__file__).parents[1] / "fixtures" / "schedule" / "tir").glob("*.py")))


def _prim_in(path: Path) -> PrimFunction:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return next(value for value in vars(loaded).values() if isinstance(value, PrimFunction))


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


@pytest.mark.parametrize("path", TIR, ids=lambda path: path.stem)
def test_tir_program_is_verified_and_canonical(path: Path) -> None:
    function = _prim_in(path)
    verify_prim_function(function)
    assert as_script(function) == path.read_text()
