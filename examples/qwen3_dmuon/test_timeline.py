"""Regressions for overlap accounting and training-only CUDA trace attribution."""

import gzip
import json

import pytest
from timeline import duration, intersection, launch_contexts, operation_label, read_rank, union


def test_concurrent_streams_are_not_double_counted():
    compute = [(0, 4), (2, 7), (8, 9), (9, 9)]
    network = [(1, 3), (5, 10), (6, 8)]
    assert union(compute) == [(0, 7), (8, 9)]
    assert duration(compute) == 8
    assert duration(network) == 7
    assert intersection(compute, network) == intersection(network, compute) == 5
    assert intersection([], network) == 0
    for bad in [[(3, 2)], [(0, float("nan"))]]:
        with pytest.raises(ValueError, match="finite and ordered"):
            union(bad)


def write_trace(folder, events):
    (folder / "rank0.json").write_text(json.dumps(dict(stage=0, data_rank=0)))
    with gzip.open(folder / "profile-rank0.json.gz", "wt") as handle:
        json.dump(dict(traceEvents=events, baseTimeNanoseconds=0, host_name="single-host"), handle)


def test_launch_window_excludes_reference_but_keeps_async_completion(tmp_path):
    events = [
        dict(cat="user_annotation", ph="X", name="training/window", ts=10, dur=10),
        dict(cat="user_annotation", ph="X", name="pipeline/forward/0/0", ts=10, dur=5),
        dict(cat="cuda_runtime", ts=9, args=dict(correlation=1)),
        dict(cat="cuda_driver", name="cuGraphLaunch", ts=12, args=dict(correlation=2)),
        dict(cat="cuda_runtime", ts=17, args=dict(correlation=3)),
        dict(cat="kernel", name="reference", ts=11, dur=9, args=dict(correlation=1)),
        dict(cat="kernel", name="graph_compute_a", ts=16, dur=4, args=dict(correlation=2)),
        dict(cat="kernel", name="graph_compute_b", ts=19, dur=4, args=dict(correlation=2)),
        dict(cat="kernel", name="ncclDevKernel_SendRecv", ts=18, dur=6, args=dict(correlation=3)),
    ]
    write_trace(tmp_path, list(reversed(events)))
    result = read_rank(tmp_path, 0)
    metrics = result["metrics"]
    assert metrics["kernel_count"] == 3
    assert metrics["compute_us"] == 7
    assert metrics["nccl_us"] == 6
    assert metrics["overlap_us"] == 5
    assert metrics["overlap_by_phase_us"]["forward"] == 5
    assert {k["phase"] for k in result["kernels"] if k["kind"] != "nccl"} == {"forward"}
    events.append(dict(cat="kernel", name="uncorrelated", ts=15, dur=1, args={}))
    write_trace(tmp_path, events)
    with pytest.raises(ValueError, match="uncorrelated"):
        read_rank(tmp_path, 0)


def test_named_operations_follow_issue_thread_and_recompute_scope():
    events = [dict(cat="user_annotation", name="qwen/layer/0", ts=0, dur=50, tid=1),
              dict(cat="user_annotation", name="communication/cp_kv", ts=2, dur=4, tid=1),
              dict(cat="user_annotation", name="dmuon.stage1_shard_reduce", ts=0, dur=20, tid=2),
              dict(cat="user_annotation", name="communication/ep_dispatch", ts=10, dur=2, tid=1),
              dict(cat="user_annotation", name="communication/ep_return", ts=30, dur=2, tid=1)]
    scopes = launch_contexts(list(reversed(events)), {1: dict(ts=3, tid=1), 2: dict(ts=15, tid=1)})
    assert operation_label("ncclAllGather", "", scopes[1], "forward") == ("CP K/V all-gather", "0")
    assert operation_label("sgemm", "aten::mm", scopes[2], "forward") == ("Expert gate/up/down GEMMs", "0")
    attention = ("autograd::engine::evaluate_function: MmBackward0", "qwen/layer/0", "aten::scaled_dot_product_attention")
    assert operation_label("flash_fwd", "", attention, "backward") == ("Recompute: SDPA attention", "0")
    assert operation_label("flash_bwd", "", (), "backward") == ("Attention backward", None)
    assert "scope unassigned" in operation_label("ncclAllGather", "", (), "forward")[0]
