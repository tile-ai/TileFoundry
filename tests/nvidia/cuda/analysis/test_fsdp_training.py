"""Read FSDP parameter residency and transfers from the existing fact layer."""

from __future__ import annotations

from math import prod
from pathlib import Path

import pytest

from tests.nvidia.cuda._fsdp_cases import FSDP_CASES, TrainingCase
from tilefoundry.analysis import MemoryMetadata, RegionMemoryMetadata, analyze
from tilefoundry.analysis.allocation import AllocationSummaryMetadata
from tilefoundry.analysis.check import check_program
from tilefoundry.cli.source import load_authored_ir
from tilefoundry.dsl import Broadcast, Partial, Split
from tilefoundry.ir.core import Call, get_metadata
from tilefoundry.ir.core.expr import SSAValue
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.sharding.reshard import Reshard
from tilefoundry.ir.types.shard_layout import shard_layout_of
from tilefoundry.ir.visitor import collect_exprs

FIXTURE = Path(__file__).parents[1] / "fixtures" / "evaluator" / "escaping_gather_view.py"


@pytest.mark.parametrize("case", FSDP_CASES, ids=lambda case: case.id)
def test_parameter_residency_and_communication_occurrences(case: TrainingCase):
    participants, retain = case.participants, case.retain
    program = case.load()
    result = analyze(*check_program(program, program.entry_function()), analysis="memory")
    report = summarize_training(result)
    assert report["persistent_parameter_and_velocity_bytes"] == 768 // participants
    assert report["all_gather"] == (2 if retain else 4) * (participants > 1)
    assert report["reduce_scatter"] == 2 * (participants > 1)
    assert report["full_weight_peak_bytes"] == ((384 if retain else 256) if participants > 1 else None)
    assert report["all_gather_result_bytes_per_participant"] == (384 if retain else 768) * (participants > 1)
    assert report["reduce_scatter_input_bytes_per_participant"] == 384 * (participants > 1)


def test_escaping_view_retains_the_full_gathered_backing_storage():
    program = load_authored_ir(f"{FIXTURE}:EscapingView")

    result = analyze(*check_program(program, program.entry_function()), analysis="memory")
    calls = [value for value in collect_exprs(result.function.body)
             if isinstance(value, Call) and isinstance(value.target, Function)]
    offsets = sorted(get_metadata(call, MemoryMetadata).offsets[0] for call in calls)
    assert len(offsets) == 2
    assert offsets[1] - offsets[0] >= 256
    assert get_metadata(result.function, RegionMemoryMetadata).peak_for("gmem").peak_bytes == 640


def summarize_training(result):
    """Read the MLP's scoped parameter facts from memory/allocation records.

    Walk each reached call occurrence. Deduplicate accepted allocation roots
    when summing full-parameter windows over lifetime boundaries.
    """
    def occurrences(function):
        for value in collect_exprs(function.body):
            if isinstance(value, Call) and isinstance(value.target, Function):
                yield from occurrences(value.target)
            elif isinstance(value, Call) and isinstance(value.target, Reshard):
                yield value

    count = {"all_gather": 0, "reduce_scatter": 0}
    gather_results = scatter_inputs = 0
    participants = result.module.effective_topologies()[0].size
    for call in occurrences(result.function):
        source = shard_layout_of(call.args[0].type.layout)
        target = shard_layout_of(call.type.layout)
        if source is None or target is None or participants == 1:
            continue
        before, after = source.attrs[0], target.attrs[0]
        if isinstance(before, Split) and isinstance(after, Broadcast):
            count["all_gather"] += 1
            gather_results += prod(call.type.shape) * call.type.dtype.bit_width // 8
        elif isinstance(before, Partial) and isinstance(after, Split):
            count["reduce_scatter"] += 1
            scatter_inputs += prod(call.args[0].type.shape) * call.args[0].type.dtype.bit_width // 8
        else:
            continue
    summary = get_metadata(result.function, AllocationSummaryMetadata)
    by_value = {value.value: value for value in summary.values}
    memory = get_metadata(result.function, RegionMemoryMetadata)
    windows = []
    for window in memory.lifetimes:
        held = window.value.leaf()
        shard = shard_layout_of(held.layout)
        if (
            shard is not None
            and all(isinstance(attr, Broadcast) for attr in shard.attrs)
            and held.shape in ((8, 8), (8, 4))
        ):
            root = summary.root(by_value[window.value])
            windows.append((root, window))
    events = {point for _, window in windows for point in (window.defined_at, window.last_used_at)}

    def live_bytes(point):
        live = {}
        for root, window in windows:
            if window.persistent or window.defined_at <= point <= window.last_used_at:
                live[root] = max(live.get(root, 0), window.bytes)
        return sum(live.values())

    persistent = sum(by_value[SSAValue(param)].bytes for param in result.function.params[2:])
    return {
        "participants": participants,
        **count,
        "full_weight_peak_bytes": max(map(live_bytes, events), default=0)
        if participants > 1
        else None,
        "persistent_parameter_and_velocity_bytes": persistent,
        "all_gather_result_bytes_per_participant": gather_results,
        "reduce_scatter_input_bytes_per_participant": scatter_inputs,
    }
