"""Two-layer FP32 training with explicit sharding and parameter residency.

``training_program`` builds HIR only. The reference evaluator runs all logical
participants on the input tensors' device; no CUDA compilation is involved.
The NVIDIA target supplies storage ownership for structural memory analysis,
not achieved bandwidth or a training-time prediction.
"""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget


def training_program(participants=2, *, retain=False):
    """Build one explicit training step with ordinary parameter/state inputs.

    Returns Y, global mean loss, local-mean dX, averaged dW1/dW2, updated
    W1/W2, and updated momentum V1/V2. Concatenated dX is divided by P when
    comparing against the derivative of the global mean loss.
    """
    if participants not in (1, 2, 4):
        raise ValueError("this example supports 1, 2 or 4 equal participants")
    target = CudaTarget("nvidia.h200_sxm", device_count=participants)
    gpu = Topology("gpu", participants)
    mesh = Mesh((gpu,), Layout((participants,), (1,)), ("dp",))
    s8 = ShardLayout.default((8, 8), mesh, (Split(0),), storage="gmem", owners=target.storage_owners)
    s4 = ShardLayout.default((8, 4), mesh, (Split(0),), storage="gmem", owners=target.storage_owners)
    b8 = ShardLayout(Layout((8, 8)), (Broadcast(),), mesh)
    b4 = ShardLayout(Layout((8, 4)), (Broadcast(),), mesh)
    local_scale = 2.0 / ((8 // participants) * 4)
    inverse_participants = 1.0 / participants

    @module(entry="retain_step" if retain else "release_step", target=target, topologies=(gpu,))
    class Training:
        @func
        def forward1(x: Tensor[(8, 8), "f32", s8], w: Tensor[(8, 8), "f32", s8]):
            full = tf.reshard(w, layout=b8, storage="gmem")
            h = tf.relu(tf.matmul(x, full))
            return h

        @func
        def forward2(h: Tensor[(8, 8), "f32", s8], w: Tensor[(8, 4), "f32", s4]):
            full = tf.reshard(w, layout=b4, storage="gmem")
            return tf.matmul(h, full)

        @func
        def retain1(x: Tensor[(8, 8), "f32", s8], w: Tensor[(8, 8), "f32", s8]):
            full = tf.reshard(w, layout=b8, storage="gmem")
            h = tf.relu(tf.matmul(x, full))
            return h, full

        @func
        def retain2(h: Tensor[(8, 8), "f32", s8], w: Tensor[(8, 4), "f32", s4]):
            full = tf.reshard(w, layout=b4, storage="gmem")
            return tf.matmul(h, full), full

        @func
        def backward2(h: Tensor[(8, 8), "f32", s8], dy: Tensor[(8, 4), "f32", s4], full: Tensor[(8, 4), "f32", b4]):
            dh = tf.matmul(dy, full, b_layout="NK")
            partial = tf.matmul(h, dy, a_layout="KM")
            gradient = tf.reshard(partial, layout=s4, storage="gmem")
            return dh, gradient * inverse_participants

        @func
        def backward1(x: Tensor[(8, 8), "f32", s8], h: Tensor[(8, 8), "f32", s8], dh: Tensor[(8, 8), "f32", s8], full: Tensor[(8, 8), "f32", b8]):
            dz = tf.where(h > 0.0, dh, 0.0)
            dx = tf.matmul(dz, full, b_layout="NK")
            partial = tf.matmul(x, dz, a_layout="KM")
            gradient = tf.reshard(partial, layout=s8, storage="gmem")
            return dx, gradient * inverse_participants

        @func
        def regather2(h: Tensor[(8, 8), "f32", s8], dy: Tensor[(8, 4), "f32", s4], w: Tensor[(8, 4), "f32", s4]):
            full = tf.reshard(w, layout=b4, storage="gmem")
            return backward2(h, dy, full)

        @func
        def regather1(x: Tensor[(8, 8), "f32", s8], h: Tensor[(8, 8), "f32", s8], dh: Tensor[(8, 8), "f32", s8], w: Tensor[(8, 8), "f32", s8]):
            full = tf.reshard(w, layout=b8, storage="gmem")
            return backward1(x, h, dh, full)

        @func
        def finish(y: Tensor[(8, 4), "f32", s4], target_value: Tensor[(8, 4), "f32", s4], dx: Tensor[(8, 8), "f32", s8], g1: Tensor[(8, 8), "f32", s8], g2: Tensor[(8, 4), "f32", s4], w1: Tensor[(8, 8), "f32", s8], w2: Tensor[(8, 4), "f32", s4], v1: Tensor[(8, 8), "f32", s8], v2: Tensor[(8, 4), "f32", s4]):
            error = y - target_value
            squared_sum = tf.reduce(error * error, axes=(0, 1), kind=ReduceKind.SUM, keepdim=False)
            loss = squared_sum / 32.0
            next_v1 = 0.9 * v1 + g1
            next_v2 = 0.9 * v2 + g2
            next_w1 = w1 - 0.01 * next_v1
            next_w2 = w2 - 0.01 * next_v2
            return y, loss, dx, g1, g2, next_w1, next_w2, next_v1, next_v2

        @func
        def release_step(x: Tensor[(8, 8), "f32", s8], target_value: Tensor[(8, 4), "f32", s4], w1: Tensor[(8, 8), "f32", s8], w2: Tensor[(8, 4), "f32", s4], v1: Tensor[(8, 8), "f32", s8], v2: Tensor[(8, 4), "f32", s4]):
            with Mesh(("gpu",), layout=(participants,), names=("dp",)) as dp:
                h = forward1(x, w1)
                y = forward2(h, w2)
                dy = (y - target_value) * local_scale
                dh, g2 = regather2(h, dy, w2)
                dx, g1 = regather1(x, h, dh, w1)
                return finish(y, target_value, dx, g1, g2, w1, w2, v1, v2)

        @func
        def retain_step(x: Tensor[(8, 8), "f32", s8], target_value: Tensor[(8, 4), "f32", s4], w1: Tensor[(8, 8), "f32", s8], w2: Tensor[(8, 4), "f32", s4], v1: Tensor[(8, 8), "f32", s8], v2: Tensor[(8, 4), "f32", s4]):
            with Mesh(("gpu",), layout=(participants,), names=("dp",)) as dp:
                h, full1 = retain1(x, w1)
                y, full2 = retain2(h, w2)
                dy = (y - target_value) * local_scale
                dh, g2 = backward2(h, dy, full2)
                dx, g1 = backward1(x, h, dh, full1)
                return finish(y, target_value, dx, g1, g2, w1, w2, v1, v2)

    return Training


def summarize_training(result):
    """Read scoped parameter facts from existing memory/allocation records.

    This example has sequential calls and no HIR loops. Traverse each reached
    call occurrence, not just each function definition. The peak below selects
    full parameter storage from the allocator's published windows and dedupes
    accepted aliases; it is not a second allocator or a total training peak.
    """
    from math import prod

    from tilefoundry.analysis import RegionMemoryMetadata
    from tilefoundry.analysis.allocation import AllocationSummaryMetadata
    from tilefoundry.ir.core import Call, get_metadata
    from tilefoundry.ir.core.expr import SSAValue
    from tilefoundry.ir.hir.function import Function
    from tilefoundry.ir.hir.sharding.reshard import Reshard
    from tilefoundry.ir.types.shard_layout import shard_layout_of
    from tilefoundry.ir.visitor import collect_exprs

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
        if shard is not None and all(isinstance(attr, Broadcast) for attr in shard.attrs) and held.shape in ((8, 8), (8, 4)):
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
        "full_weight_peak_bytes": max(map(live_bytes, events), default=0) if participants > 1 else None,
        "persistent_parameter_and_velocity_bytes": persistent,
        "all_gather_result_bytes_per_participant": gather_results,
        "reduce_scatter_input_bytes_per_participant": scatter_inputs,
    }


if __name__ == "__main__":
    import argparse
    import json

    from tilefoundry.analysis import analyze
    from tilefoundry.analysis.check import check_program

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--participants", type=int, choices=(1, 2, 4), default=2)
    args = parser.parse_args()
    for retain in (False, True):
        program = training_program(args.participants, retain=retain)
        analysis = analyze(*check_program(program, program.entry_function()), analysis="memory")
        print(json.dumps({"policy": "retain" if retain else "release", **summarize_training(analysis)}))
