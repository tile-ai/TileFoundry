"""Capture consecutive training steps after the numerical reference checks."""

import time

import torch
import torch.distributed as dist


def capture(step, *, first, warmup, steps, output, rank):
    """Measure an unprofiled window, then capture a separate CUDA timeline."""
    for index in range(first, first + warmup):
        step(index)
    first += warmup
    dist.barrier()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for index in range(first, first + steps):
        step(index)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    measurements = [None] * dist.get_world_size()
    dist.all_gather_object(measurements, elapsed)
    first += steps
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
    return dict(trace=str(trace), warmup_steps=warmup, measured_steps=steps, profiled_steps=steps,
                unprofiled_rank_seconds=measurements, unprofiled_window_seconds=max(measurements),
                unprofiled_seconds_per_step=max(measurements) / steps,
                training_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                includes="forward, backward, optimizer and communication; no reference work",
                synchronization="world barrier and device sync at window boundaries; none between steps")
