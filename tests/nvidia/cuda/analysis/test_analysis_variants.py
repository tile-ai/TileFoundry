"""Analysis inputs with changed targets, options, or solver behavior."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from tests.core._corpus import load_program
from tests.nvidia.cuda.analysis._contracts import (
    _footprint_bytes,
    _reuse_conclusions,
    _working_set_bytes,
)
from tests.nvidia.cuda.fixtures.placed import persistent_gemm_tiled
from tests.nvidia.cuda.fixtures.placed.gemm_schedules import (
    Gemm_MNK_NN128x128x64_w12x11_k16384,
)
from tests.nvidia.cuda.fixtures.placed.hand_checked import (
    CapacityExceeded,
    WaveTruncation,
)
from tests.nvidia.cuda.fixtures.placed.persistent_gemm_tiled import PersistentGemmTiled
from tilefoundry.analysis import (
    AnalysisPrecision,
    allocation,
    analyze,
)
from tilefoundry.analysis import footprint as footprint_analysis
from tilefoundry.analysis.check import check_program
from tilefoundry.analysis.errors import AnalysisError
from tilefoundry.analysis.report import report_data
from tilefoundry.ir.hir.specialize import SpecializationError
from tilefoundry_nvidia import CudaTarget


def _report(module, *, analysis=("memory",)) -> dict:
    result = analyze(*check_program(module, module.entry_function()), analysis=analysis)
    data = report_data(
        module=result.module,
        function=result.function,
        analyses=result.analyses,
        topology_level=result.topology_level,
        executed=result.executed,
        metadata_types=result.metadata_types,
    )
    return data


def _memory_record(module) -> dict:
    return _report(module)["functions"][0]["function_records"]["memory"]


@pytest.mark.parametrize("precision", tuple(AnalysisPrecision))
def test_persistent_gemm_over_reuse_conclusions_match_the_reviewed_report(
    precision: AnalysisPrecision,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(footprint_analysis, "counted_precision", lambda items: precision)
    memory = _memory_record(Gemm_MNK_NN128x128x64_w12x11_k16384)

    assert _reuse_conclusions(memory) == [
        {
            "buffer": "b",
            "time": "mi",
            "space": "cta.x",
            "holds_bytes": 184_549_376,
            "reuse_bytes": 4_697_620_480,
            "fits": False,
        },
        {
            "buffer": "a",
            "time": "ni",
            "space": "cta.y",
            "holds_bytes": 96_468_992,
            "reuse_bytes": 1_610_612_736,
            "fits": False,
        },
    ]
    category = (
        "errors"
        if precision in (AnalysisPrecision.EXACT, AnalysisPrecision.LOWER_BOUND)
        else "advisories"
    )
    assert all(row["precision"] == precision.value for row in memory["reuse_windows"])
    assert memory[category] == [
        "l2 reuse window mi holds 176.00MiB at a 132-unit wave, exceeding capacity 47.68MiB",
        "l2 reuse window ni holds 92.00MiB at a 132-unit wave, exceeding capacity 47.68MiB",
    ]
    assert memory["advisories" if category == "errors" else "errors"] == []


@pytest.mark.parametrize("precision", tuple(AnalysisPrecision))
def test_capacity_exceeded_matches_the_written_ratio(
    precision: AnalysisPrecision,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(footprint_analysis, "counted_precision", lambda items: precision)
    data = _report(CapacityExceeded)
    memory = data["functions"][0]["function_records"]["memory"]
    used = _working_set_bytes(memory)
    capacity = 1_048_576
    category = (
        "errors"
        if precision in (AnalysisPrecision.EXACT, AnalysisPrecision.LOWER_BOUND)
        else "advisories"
    )

    assert (used, capacity, used * 100 / capacity) == (1_572_864, 1_048_576, 150.0)
    diagnosis = [
        "l2 working set 1.50MiB at the first iteration of a 1-unit wave exceeds capacity 1.00MiB"
    ]
    assert memory["errors"] == ["rmem placement peak 1.50MiB exceeds capacity 256.00KiB"] + (
        diagnosis if category == "errors" else []
    )
    assert memory["advisories"] == (diagnosis if category == "advisories" else [])


def test_runtime_loop_start_names_the_unbounded_value() -> None:
    module = load_program(Path(__file__).parents[1] / "fixtures/diagnostics/analysis_runtime_start.py")
    with pytest.raises(
        AnalysisError,
        match=r"loop 'n' takes its start from 'start'.*analysis needs a literal start or a stated value range",
    ):
        _memory_record(module)


def test_symbolic_divisor_loop_start_does_not_block_typing() -> None:
    module = load_program(Path(__file__).parents[1] / "fixtures/logical/symbolic_loop_start.py")
    with pytest.raises(SpecializationError, match=r"still states \['chunks'\] as ranges"):
        check_program(module, module.entry_function())
    check_program(module, module.entry_function(), dims={"chunks": 8})


def test_wave_truncation_counts_only_the_resident_ctas() -> None:
    wide_target = CudaTarget(
        replace(WaveTruncation.target.device, sm_count=256),
        architecture=WaveTruncation.target.architecture,
    )
    all_declared_data = _report(
        replace(WaveTruncation, target=wide_target), analysis=("memory", "roofline")
    )
    all_declared = all_declared_data["functions"][0]["function_records"]["memory"]

    assert all_declared["traffic"]["storage"]["gmem"]["total"] == {"read": 2048, "write": 0}
    assert _footprint_bytes(all_declared, "x") == 256 * 4 * 2
    assert all_declared["footprint"]["buffers"]["x"]["gmem"]["total"] == 2048


def test_persistent_tiled_unseeded(monkeypatch: pytest.MonkeyPatch) -> None:
    """The footprint is held at the loop start; one tile product is its own size.

    Each step multiplies an ``(BM, BK)`` by a ``(BK, BN)`` bf16 tile into an f32
    ``(BM, BN)`` register tile: ``BM * BN * 4`` bytes written and
    ``2 * BM * BN * BK`` bf16 flops by the CTA that runs it, whatever layout the
    operands were staged in. Shared memory holds only the two operand tiles, so
    placing them needs their sum and no more. A seed nobody could build bounds
    nothing: the solver still finds a placement.
    """
    monkeypatch.setattr(allocation, "_first_fit", lambda *args, **kwargs: None)
    data = _report(PersistentGemmTiled, analysis=("memory", "compute-cost"))
    memory = data["functions"][0]["function_records"]["memory"]
    assert _footprint_bytes(memory, "a") == 12 * 64 * 32 * 2
    assert _footprint_bytes(memory, "b") == 11 * 32 * 64 * 2
    assert memory["footprint"]["precision"] == "exact"
    (product,) = (
        call
        for call in data["functions"][0]["calls"]
        if [operand["type"] for operand in call["memory"]["operands"]]
        == [
            f"bf16[{persistent_gemm_tiled.BM},{persistent_gemm_tiled.BK}] smem",
            f"bf16[{persistent_gemm_tiled.BK},{persistent_gemm_tiled.BN}] smem",
            f"f32[{persistent_gemm_tiled.BM},{persistent_gemm_tiled.BN}] rmem",
        ]
    )
    tile_bytes = persistent_gemm_tiled.BM * persistent_gemm_tiled.BN * 4
    assert product["memory"]["operands"][-1]["write"] == tile_bytes
    flops = product["compute-cost"]["flops"]["bf16"]["per_unit"]
    assert flops == [
        2 * persistent_gemm_tiled.BM * persistent_gemm_tiled.BN * persistent_gemm_tiled.BK
    ]
    operands = (
        persistent_gemm_tiled.BM * persistent_gemm_tiled.BK * 2,
        persistent_gemm_tiled.BK * persistent_gemm_tiled.BN * 2,
    )
    peaks = {peak["memory_level"]: peak["peak_bytes"] for peak in memory["peaks"]}
    assert peaks["smem"] >= sum(operands)
