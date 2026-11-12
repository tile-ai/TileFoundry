"""Capture consecutive training steps after the numerical reference checks."""

import statistics
import time

import torch
import torch.distributed as dist


def capture(step, *, first, warmup, steps, output, rank, rounds=1, timing_steps=None):
    """Measure an unprofiled window, then capture a separate CUDA timeline."""
    for index in range(first, first + warmup):
        step(index)
    first += warmup
    timed = timing_steps or steps
    torch.cuda.reset_peak_memory_stats()
    first_timed = first
    samples = []
    for _ in range(rounds):
        dist.barrier()
        torch.cuda.synchronize()
        started = time.perf_counter()
        for index in range(first, first + timed):
            step(index)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        measurements = [None] * dist.get_world_size()
        dist.all_gather_object(measurements, elapsed)
        samples.append(measurements)
        first += timed
    peak = torch.cuda.max_memory_allocated()
    per_step = [max(sample) / timed for sample in samples]
    trace = output / f"profile-rank{rank}.json.gz"
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                           torch.profiler.ProfilerActivity.CUDA]) as profiler:
        dist.barrier()
        torch.cuda.synchronize()
        with torch.profiler.record_function("training/window"):
            for index in range(first, first + steps):
                with torch.profiler.record_function(f"training/step/{index}"):
                    step(index)
            torch.cuda.synchronize()
    profiler.export_chrome_trace(str(trace))
    return dict(trace=str(trace), warmup_steps=warmup, measured_steps=rounds * timed, profiled_steps=steps,
                timing_rounds=rounds, steps_per_round=timed, rank_window_seconds=samples,
                first_timed_step=first_timed, first_profiled_step=first, next_step=first + steps,
                samples_seconds_per_step=per_step,
                unprofiled_rank_seconds=list(map(sum, zip(*samples))),
                unprofiled_window_seconds=sum(map(max, samples)),
                unprofiled_seconds_per_step=statistics.median(per_step),
                training_peak_allocated_bytes=peak,
                includes="batch preparation, forward, backward, optimizer and communication; no reference work",
                synchronization="world barrier and device sync at window boundaries; none between steps")
