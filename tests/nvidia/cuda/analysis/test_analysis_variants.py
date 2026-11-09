"""Analysis inputs with changed targets, options, or solver behavior."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from tests.nvidia.cuda.analysis._contracts import (
    _footprint_bytes,
    _reuse_conclusions,
    _working_set_bytes,
)
from tilefoundry import analysis as tilefoundry_analysis
from tilefoundry import cli, inspection
from tilefoundry.analysis import allocation, reach
from tilefoundry.interfaces import valuation
from tilefoundry_nvidia import cuda

PLACED = Path(__file__).parents[1] / "fixtures" / "placed"


def _report(module, *, analysis=("memory",)) -> dict:
    result = tilefoundry_analysis.analyze(*tilefoundry_analysis.check_program(module, module.entry_function()), analysis=analysis)
    data = inspection.report_data(
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


@pytest.mark.parametrize("precision", tuple(tilefoundry_analysis.Precision))
def test_persistent_gemm_reports_reuse_overflow(
    precision: tilefoundry_analysis.Precision,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(reach, "counted_precision", lambda items: precision)
    memory = _memory_record(
        cli.load_authored_ir(f"{PLACED / 'gemm_schedules.py'}:Gemm_MNK_NN128x128x64_w12x11_k16384")
    )

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
        if precision in (tilefoundry_analysis.Precision.EXACT, tilefoundry_analysis.Precision.LOWER_BOUND)
        else "advisories"
    )
    assert all(row["precision"] == precision.value for row in memory["reuse_windows"])
    assert memory[category] == [
        "l2 reuse window mi holds 176.00MiB at a 132-unit wave, exceeding capacity 47.68MiB",
        "l2 reuse window ni holds 92.00MiB at a 132-unit wave, exceeding capacity 47.68MiB",
    ]
    assert memory["advisories" if category == "errors" else "errors"] == []


@pytest.mark.parametrize("precision", tuple(tilefoundry_analysis.Precision))
def test_working_set_overflow_reported(
    precision: tilefoundry_analysis.Precision,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(reach, "counted_precision", lambda items: precision)
    data = _report(cli.load_authored_ir(f"{PLACED / 'hand_checked.py'}:CapacityExceeded"))
    memory = data["functions"][0]["function_records"]["memory"]
    used = _working_set_bytes(memory)
    capacity = 1_048_576
    category = (
        "errors"
        if precision in (tilefoundry_analysis.Precision.EXACT, tilefoundry_analysis.Precision.LOWER_BOUND)
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


def test_runtime_loop_start_names_value() -> None:
    module = cli.load_authored_ir(
        str(Path(__file__).parents[1] / "fixtures/diagnostics/analysis_runtime_start.py")
    )
    with pytest.raises(
        tilefoundry_analysis.Error,
        match=r"loop 'n' takes its start from 'start'.*analysis needs a literal start or a stated value range",
    ):
        _memory_record(module)


def test_bound_symbolic_loop_start_typechecks() -> None:
    module = cli.load_authored_ir(
        str(Path(__file__).parents[1] / "fixtures/logical/symbolic_loop_start.py")
    )
    with pytest.raises(valuation.SpecializationError, match=r"still states \['chunks'\] as ranges"):
        tilefoundry_analysis.check_program(module, module.entry_function())
    tilefoundry_analysis.check_program(module, module.entry_function(), dims={"chunks": 8})


def test_wave_truncation_counts_resident_ctas() -> None:
    WaveTruncation = cli.load_authored_ir(f"{PLACED / 'hand_checked.py'}:WaveTruncation")
    wide_target = cuda.Target(
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
    monkeypatch.setattr(allocation, "first_fit", lambda *args, **kwargs: None)
    persistent_gemm_tiled = PLACED / "persistent_gemm_tiled.py"
    constants = cli.load_namespace(str(persistent_gemm_tiled))[0]
    BM, BN, BK = constants["BM"], constants["BN"], constants["BK"]
    PersistentGemmTiled = cli.load_authored_ir(f"{persistent_gemm_tiled}:PersistentGemmTiled")
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
            f"bf16[{BM},{BK}] smem",
            f"bf16[{BK},{BN}] smem",
            f"f32[{BM},{BN}] rmem",
        ]
    )
    tile_bytes = BM * BN * 4
    assert product["memory"]["operands"][-1]["write"] == tile_bytes
    flops = product["compute-cost"]["flops"]["bf16"]["per_unit"]
    assert flops == [2 * BM * BN * BK]
    operands = (BM * BK * 2, BK * BN * 2)
    peaks = {peak["memory_level"]: peak["peak_bytes"] for peak in memory["peaks"]}
    assert peaks["smem"] >= sum(operands)
