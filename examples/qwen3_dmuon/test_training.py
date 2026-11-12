"""Protect matched training comparisons and the timing/profiling boundary."""

from contextlib import nullcontext
from copy import deepcopy
from types import SimpleNamespace

import pytest
from benchmark_training import check_pair
from parallel import training_groups
from profiling import capture


def test_repeated_windows_preserve_consecutive_steps_and_exclude_profile_peak(monkeypatch, tmp_path):
    import profiling

    clock = iter((0, 8, 10, 22, 25, 29))
    monkeypatch.setattr(profiling.time, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(profiling.dist, "barrier", lambda: None)
    monkeypatch.setattr(profiling.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(profiling.dist, "all_gather_object", lambda result, value: result.__setitem__(slice(None), [value, value * 2]))
    monkeypatch.setattr(profiling.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(profiling.torch.cuda, "reset_peak_memory_stats", lambda: None)
    calls = []
    monkeypatch.setattr(profiling.torch.cuda, "max_memory_allocated", lambda: len(calls))
    monkeypatch.setattr(profiling.torch.profiler, "profile", lambda **kwargs: nullcontext(SimpleNamespace(export_chrome_trace=lambda path: None)))
    monkeypatch.setattr(profiling.torch.profiler, "record_function", lambda name: nullcontext())
    result = capture(calls.append, first=2, warmup=2, steps=2, timing_steps=4, rounds=3, output=tmp_path, rank=0)
    assert calls == list(range(2, 18))
    assert result["first_timed_step"] == 4
    assert result["first_profiled_step"] == 16
    assert result["next_step"] == 18
    assert result["samples_seconds_per_step"] == [4, 6, 2]
    assert result["unprofiled_seconds_per_step"] == 4
    assert result["training_peak_allocated_bytes"] == 14


def test_comparison_rejects_changed_work_or_sharding():
    baseline = dict(config=dict(cp=2, ep=1, tokens=8192), valid_targets_per_step=65000,
                    warmup_steps=2, steps_per_round=4, profile_steps=2, numerical_steps=2,
                    samples_seconds_per_step=[1, 1.1, 0.9])
    candidate = deepcopy(baseline)
    candidate["samples_seconds_per_step"] = [0.8, 0.9, 0.7]
    check_pair(baseline, candidate)
    for key, value in (("config", dict(cp=1, ep=2, tokens=8192)), ("valid_targets_per_step", 32000),
                       ("samples_seconds_per_step", [0.8])):
        broken = candidate | {key: value}
        with pytest.raises(ValueError, match="paired candidates"):
            check_pair(baseline, broken)


def test_hsdp_axes_stay_inside_the_pipeline_stage(monkeypatch):
    import torch.distributed as dist

    monkeypatch.setattr(dist, "get_world_size", lambda: 8)
    monkeypatch.setattr(dist, "get_rank", lambda: 5)
    membership = []
    monkeypatch.setattr(dist, "new_group", lambda ranks: membership.append(tuple(ranks)) or tuple(ranks))
    groups = training_groups(2, 2)
    assert groups.pipeline == (1, 5)
    assert groups.data == (4, 5, 6, 7)
    assert groups.shard == (4, 5)
    assert groups.replicate == (5, 7)
    assert groups.ranks.tolist() == [[4, 5], [6, 7]]
    assert groups.size == 4 and groups.shards == 2
