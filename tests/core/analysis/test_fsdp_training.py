"""Read FSDP parameter residency and transfers from the existing fact layer."""

from __future__ import annotations

import pytest

from examples.fsdp_training import summarize_training, training_program
from tilefoundry.analysis import MemoryMetadata, RegionMemoryMetadata, analyze
from tilefoundry.analysis.check import check_program
from tilefoundry.dsl import (
    Broadcast,
    Layout,
    Mesh,
    ShardLayout,
    Split,
    Tensor,
    Topology,
    func,
    module,
    tf,
)
from tilefoundry.ir.core import Call, get_metadata
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.visitor import collect_exprs
from tilefoundry_nvidia import CudaTarget


@pytest.mark.parametrize("participants", (1, 2, 4))
@pytest.mark.parametrize("retain", (False, True), ids=("release", "retain"))
def test_parameter_residency_and_communication_occurrences(participants, retain):
    program = training_program(participants, retain=retain)
    result = analyze(*check_program(program, program.entry_function()), analysis="memory")
    report = summarize_training(result)
    assert report["persistent_parameter_and_velocity_bytes"] == 768 // participants
    assert report["all_gather"] == (2 if retain else 4) * (participants > 1)
    assert report["reduce_scatter"] == 2 * (participants > 1)
    assert report["full_weight_peak_bytes"] == ((384 if retain else 256) if participants > 1 else None)
    assert report["all_gather_result_bytes_per_participant"] == (384 if retain else 768) * (participants > 1)
    assert report["reduce_scatter_input_bytes_per_participant"] == 384 * (participants > 1)


def test_escaping_view_retains_the_full_gathered_backing_storage():
    gpu = Topology("gpu", 2)
    mesh = Mesh((gpu,), Layout((2,), (1,)), ("dp",))
    split = ShardLayout(Layout((8, 8)), (Split(0),), mesh)
    full = ShardLayout(Layout((8, 8)), (Broadcast(),), mesh)

    @module(entry="run", target=CudaTarget("nvidia.h200_sxm", device_count=2), topologies=(gpu,))
    class EscapingView:
        @func
        def stage(w: Tensor[(8, 8), "f32", split]):
            gathered = tf.reshard(w, layout=full, storage="gmem")
            return tf.slice(gathered, (0, 0), sizes=(4, 8), strides=(1, 1))

        @func
        def run(w: Tensor[(8, 8), "f32", split]):
            with Mesh(("gpu",), layout=(2,), names=("dp",)) as _dp:
                view = stage(w)  # noqa: F821
                other = stage(w)  # noqa: F821
                return view + other

    result = analyze(*check_program(EscapingView, EscapingView.entry_function()), analysis="memory")
    calls = [value for value in collect_exprs(result.function.body)
             if isinstance(value, Call) and isinstance(value.target, Function)]
    offsets = sorted(get_metadata(call, MemoryMetadata).offsets[0] for call in calls)
    assert len(offsets) == 2
    assert offsets[1] - offsets[0] >= 256
    assert get_metadata(result.function, RegionMemoryMetadata).peak_for("gmem").peak_bytes == 640
