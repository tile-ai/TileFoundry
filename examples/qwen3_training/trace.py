"""Observe execution contracts, capture training and attribute device operations."""

import gzip
import json
import statistics
import time
from bisect import bisect_right
from collections import Counter, defaultdict
from math import isfinite

import torch
import torch.distributed as dist
from torch.distributed.pipelining import PipelineStage

from tilefoundry import core
from tilefoundry.interfaces import evaluator, typeinfer
from tilefoundry.ir import core as ir_core
from tilefoundry.ir import expr, types
from tilefoundry.ir.hir import sharding


def collective_function(op, shape, dtype):
    """Infer the primitive once through the ordinary HIR function interface."""
    type_ = types.TensorType.from_shape(tuple(shape), dtype=evaluator.from_torch_dtype(dtype))
    source = expr.Var("payload", type=type_)
    call = expr.Call(op, (source,), type=type_)
    typeinfer.TypeInferVisitor().visit(call, typeinfer.TypeInferContext())
    return ir_core.Function.from_parts(name="communicate", params=(source,), body=call, return_type=call.type)

def exchange_contracts(mesh, shape, tokens, dtype, contexts):
    """Model the hidden and router payload at each pipeline cut in both directions."""
    members = types.Placement.from_mesh(mesh)
    stages = len(members)
    contracts = {}
    for cut in range(1, stages):
        payload_shapes = ((1, tokens, shape.hidden),
                          (1, cut * shape.layers // stages, tokens, shape.experts))
        for direction, src, dst in (("fwd", cut - 1, cut), ("bwd", cut, cut - 1)):
            for slot, payload_shape in enumerate(payload_shapes):
                count = torch.Size(payload_shape).numel()
                counts = tuple(tuple(count if (i, j) == (src, dst) else 0 for j in range(stages))
                               for i in range(stages))
                op = sharding.Exchange(participants=mesh, context=contexts[direction], counts=counts)
                function = collective_function(op, (count,), dtype)
                sample = torch.arange(count, dtype=torch.float32).to(dtype)
                source = core.ShardTensor(function.params[0].type, ((members[src], sample),))
                result = evaluator.evaluate(function, source)
                torch.testing.assert_close(result.to_local(members[dst]), sample, rtol=0, atol=0)
                key = direction, members[src].ids["gpu"], members[dst].ids["gpu"], slot
                contracts[key] = dict(shape=payload_shape, count=count, dtype=str(dtype),
                                      counts=counts, context=op.context,
                                      participants=tuple(p.ids["gpu"] for p in members))
    return contracts

class TracedStage(PipelineStage):
    """Observe scheduler endpoints without replacing PyTorch's transport or schedule."""

    def __init__(self, *args, mesh, shape, tokens, **kwargs):
        super().__init__(*args, **kwargs)
        contexts = {direction: f"nccl/{group.group_name}" for direction, group in (
            ("fwd", getattr(self, "_downstream_group", self.group)),
            ("bwd", getattr(self, "_upstream_group", self.group)))}
        self.contracts = exchange_contracts(mesh, shape, tokens, torch.float32, contexts)
        self.transfers = []
        self.execution = []
        self.iteration = 0

    def record(self, operations, direction, action, microbatch):
        for slot, op in enumerate(operations):
            src, dst = (dist.get_rank(), op.peer) if action == "send" else (op.peer, dist.get_rank())
            expected = self.contracts[direction, src, dst, slot]
            assert tuple(dist.get_process_group_ranks(op.group)) == expected["participants"]
            assert f"nccl/{op.group.group_name}" == expected["context"]
            assert tuple(op.tensor.shape) == expected["shape"], (op.tensor.shape, expected)
            assert str(op.tensor.dtype) == expected["dtype"], (op.tensor.dtype, expected)
            self.transfers.append(dict(step=self.iteration, direction=direction, action=action,
                                       microbatch=microbatch, src=src, dst=dst, slot=slot,
                                       count=op.tensor.numel(), dtype=str(op.tensor.dtype),
                                       context=expected["context"]))
        return operations

    def get_fwd_send_ops(self, chunk):
        return self.record(super().get_fwd_send_ops(chunk), "fwd", "send", chunk)

    def get_fwd_recv_ops(self, chunk):
        return self.record(super().get_fwd_recv_ops(chunk), "fwd", "recv", chunk)

    def get_bwd_send_ops(self, chunk):
        return self.record(super().get_bwd_send_ops(chunk), "bwd", "send", chunk)

    def get_bwd_recv_ops(self, chunk):
        return self.record(super().get_bwd_recv_ops(chunk), "bwd", "recv", chunk)

    def forward_one_chunk(self, chunk, *args, **kwargs):
        if hasattr(self, "semantic_model"):
            self.semantic_model.microbatch = chunk
        self.execution.append(dict(step=self.iteration, action="forward", microbatch=chunk))
        with torch.profiler.record_function(f"pipeline/forward/{self.stage_index}/{chunk}"):
            return super().forward_one_chunk(chunk, *args, **kwargs)

    def backward_one_chunk(self, chunk, *args, **kwargs):
        if hasattr(self, "semantic_model"):
            self.semantic_model.microbatch = chunk
        self.execution.append(dict(step=self.iteration, action="backward", microbatch=chunk))
        with torch.profiler.record_function(f"pipeline/backward/{self.stage_index}/{chunk}"):
            return super().backward_one_chunk(chunk, *args, **kwargs)

def check_endpoints(traces, expected_transfers):
    """Require unique matching payloads in FIFO order on each directed context."""
    sent, received = {}, {}
    for trace in traces:
        for event in trace:
            key = tuple(event[name] for name in ("step", "direction", "microbatch", "src", "dst", "slot", "count", "dtype", "context"))
            channel = event["context"], event["src"], event["dst"]
            (sent if event["action"] == "send" else received).setdefault(channel, []).append(key)
    assert sent == received, "send/receive endpoint or FIFO order mismatch"
    assert all(len(events) == len(set(events)) for events in sent.values()), "duplicate transfer"
    matched = sum(map(len, sent.values()))
    assert matched == expected_transfers, (matched, expected_transfers)
    return matched

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
                                           torch.profiler.ProfilerActivity.CUDA],
                                record_shapes=True, profile_memory=True) as profiler:
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

def union(intervals):
    """Merge overlapping device intervals without summing concurrent streams twice."""
    merged = []
    for start, end in sorted(intervals):
        if not isfinite(start) or not isfinite(end) or end < start:
            raise ValueError("interval endpoints must be finite and ordered")
        if merged and start <= merged[-1][1]:
            merged[-1] = merged[-1][0], max(end, merged[-1][1])
        elif end > start:
            merged.append((start, end))
    return merged

def duration(intervals):
    return sum(end - start for start, end in union(intervals))

def intersection(left, right):
    """Measure time when both sets have at least one active CUDA kernel."""
    left, right = union(left), union(right)
    i = j = 0
    total = 0.0
    while i < len(left) and j < len(right):
        total += max(0, min(left[i][1], right[j][1]) - max(left[i][0], right[j][0]))
        if left[i][1] < right[j][1]:
            i += 1
        else:
            j += 1
    return total

def launch_contexts(events, launches):
    """Resolve thread-local ranges at issue time, including autograd and CP/EP scopes."""
    import heapq

    threads = defaultdict(list)
    dispatch = defaultdict(list)
    for event in events:
        name, category = event.get("name", ""), event.get("cat")
        keep = category == "user_annotation" and name.startswith(("qwen/", "communication/", "dmuon.", "FSDP", "nccl:"))
        keep |= category == "cpu_op" and (name.startswith(("autograd::engine", "c10d::")) or "scaled_dot_product" in name)
        if keep:
            threads[event.get("tid")].append((event["ts"], 0, len(threads[event.get("tid")]), event))
        if name in ("communication/ep_dispatch", "communication/ep_return"):
            dispatch[event.get("tid")].append(event)
    for thread, boundaries in dispatch.items():
        start = None
        for event in sorted(boundaries, key=lambda e: e["ts"]):
            if event["name"] == "communication/ep_dispatch":
                start = event["ts"] + event["dur"]
            elif start is not None and start < event["ts"]:
                scope = dict(name="experts/local_compute", ts=start, dur=event["ts"]-start)
                threads[thread].append((start, 0, len(threads[thread]), scope))
                start = None
    for correlation, event in launches.items():
        threads[event.get("tid")].append((event["ts"], 1, correlation, event))
    contexts = {}
    for points in threads.values():
        active, ends = {}, []
        for when, kind, identity, event in sorted(points, key=lambda item: item[:3]):
            while ends and ends[0][0] <= when:
                _, ended = heapq.heappop(ends)
                active.pop(ended, None)
            if kind == 0:
                active[identity] = event
                heapq.heappush(ends, (when + event["dur"], identity))
            else:
                contexts[identity] = tuple(e["name"] for e in sorted(active.values(), key=lambda e: e["ts"]))
    return contexts

def operation_label(kernel, operator, scopes, phase):
    """Name observed operations; unassigned communication is kept explicit."""
    context = " ".join(scopes)
    nccl = "nccl" in kernel.lower()
    layer = next((name.rsplit("/", 1)[1] for name in reversed(scopes) if name.startswith("qwen/layer/")), None)
    if nccl:
        for scope, label in (("communication/cp_kv", "CP K/V all-gather"),
                             ("communication/ep_counts", "EP count all-gather"),
                             ("communication/ep_dispatch", "EP token/ID dispatch"),
                             ("communication/ep_return", "EP output return"),
                             ("_AllGatherBackward", "CP gradient reduce-scatter"),
                             ("_AlltoAllSingleBackward", "EP gradient all-to-all"),
                             ("dmuon.stage1_shard_reduce", "DMuon owner-gradient reduction"),
                             ("dmuon.replicate_broadcast", "HSDP parameter publication")):
            if scope in context:
                return label, layer
        if "nccl:coalesced" in context and "SendRecv" in kernel:
            return "Grouped P2P send/recv", layer
        if any(name.startswith("dmuon.") for name in scopes):
            return "DMuon parameter publication" if "Broadcast" in kernel or "Gather" in kernel else "DMuon gradient communication", layer
        if any(name.startswith(("c10d::send", "c10d::recv", "c10d::isend", "c10d::irecv")) for name in scopes):
            return "PP activation/gradient transfer", layer
        family = next((word for word in ("ReduceScatter", "AllGather", "AllReduce", "Broadcast", "Reduce", "SendRecv") if word in kernel), "collective")
        return f"NCCL {family} (scope unassigned)", layer
    if "scaled_dot_product" in context.lower() or any(word in kernel.lower() for word in ("flash_fwd", "flash_bwd", "attention")):
        backward = any(("scaled_dot_product" in name.lower() or "scaleddotproduct" in name.lower()) and "backward" in name.lower() for name in scopes)
        label = "Attention backward" if backward or "bwd" in kernel.lower() else "Attention forward"
    elif operator in ("aten::mm", "aten::bmm", "aten::addmm") or "gemm" in kernel.lower():
        label = ("Muon update GEMMs" if phase == "optimizer" else "Expert gate/up/down GEMMs" if "experts/local_compute" in context
                 else "Backward GEMMs" if phase == "backward" and layer is None else "Projection GEMMs")
    elif "embedding" in operator or "embedding" in kernel.lower():
        label = "Embedding lookup/gradient"
    elif "softmax" in operator.lower() or "softmax" in kernel.lower() or "nll_loss" in operator:
        label = "Router/loss softmax"
    elif any(word in operator for word in ("index", "scatter", "gather", "sort", "bincount")):
        label = "Token routing/indexing"
    elif phase == "optimizer":
        label = "Momentum/AdamW/pointwise update"
    else:
        label = "Pointwise/normalization/packing"
    if phase == "backward" and layer is not None:
        label = "Recompute: " + label
    return label, layer

def read_rank(folder, rank):
    """Correlate kernels, including CUDA graph replay, to host launch ranges."""
    manifest = json.loads((folder / f"rank{rank}.json").read_text())
    with gzip.open(folder / f"profile-rank{rank}.json.gz", "rt") as handle:
        trace = json.load(handle)
    events = trace["traceEvents"]
    annotations = [e for e in events if e.get("cat") == "user_annotation" and e.get("ph") == "X"]
    windows = [(e["ts"], e["ts"] + e["dur"]) for e in annotations if e["name"] == "training/window"]
    if len(windows) != 1:
        raise ValueError("expected one recorded training window")
    phases = sorted((e for e in annotations if e["name"].startswith("pipeline/")
                     or e["name"] == "training/optimizer"), key=lambda e: e["ts"])
    starts = [e["ts"] for e in phases]
    steps = sorted((e for e in annotations if e["name"].startswith("training/step/")), key=lambda e: e["ts"])
    step_starts = [e["ts"] for e in steps]
    launches = {e.get("args", {}).get("correlation"): e for e in events
                if e.get("cat") in {"cuda_runtime", "cuda_driver"} and "correlation" in e.get("args", {})}
    contexts = launch_contexts(events, launches)
    operators = {e.get("args", {}).get("External id"): e["name"] for e in events if e.get("cat") == "cpu_op"}
    kernels, copies, missing = [], [], 0
    for event in events:
        category = event.get("cat")
        if category not in ("kernel", "gpu_memcpy", "gpu_memset"):
            continue
        launch = launches.get(event.get("args", {}).get("correlation"))
        if launch is None:
            missing += category == "kernel"
            continue
        if not windows[0][0] <= launch["ts"] <= windows[0][1]:
            continue
        index = bisect_right(starts, launch["ts"]) - 1
        phase = "other"
        if index >= 0 and launch["ts"] < phases[index]["ts"] + phases[index]["dur"]:
            name = phases[index]["name"]
            phase = "optimizer" if name == "training/optimizer" else name.split("/")[1]
        kind = "nccl" if "nccl" in event["name"].lower() else phase
        step_index = bisect_right(step_starts, launch["ts"]) - 1
        step = int(steps[step_index]["name"].rsplit("/", 1)[1]) if step_index >= 0 else None
        operator = operators.get(launch.get("args", {}).get("External id"), "")
        operation, layer = operation_label(event["name"], operator, contexts.get(event.get("args", {}).get("correlation"), ()), phase)
        microbatch = phases[index]["name"].rsplit("/", 1)[-1] if index >= 0 and phases[index]["name"].startswith("pipeline/") else None
        target = kernels if category == "kernel" else copies
        target.append(dict(event, kind=kind, phase=phase, step=step, operation=operation, layer=layer,
                           operator=operator, microbatch=microbatch, launch_ts=launch["ts"],
                           scopes=contexts.get(event.get("args", {}).get("correlation"), ())))
    if missing or not kernels:
        raise ValueError(f"incomplete CUDA trace: {missing} uncorrelated kernels, {len(kernels)} attributed")
    by_kind = defaultdict(list)
    for event in kernels:
        by_kind[event["kind"]].append((event["ts"], event["ts"] + event["dur"]))
    communication = by_kind["nccl"]
    compute = [interval for kind, intervals in by_kind.items() if kind != "nccl" for interval in intervals]
    overlap = intersection(compute, communication)
    communication_us = duration(communication)
    metrics = dict(rank=rank, stage=manifest["stage"], data_rank=manifest["data_rank"],
                   compute_us=duration(compute), nccl_us=communication_us, overlap_us=overlap,
                   nccl_without_compute_us=communication_us - overlap,
                   nccl_overlap_fraction=overlap / communication_us if communication_us else 0,
                   compute_overlap_fraction=overlap / duration(compute) if duration(compute) else 0,
                   overlap_by_phase_us={kind: intersection(intervals, communication)
                                        for kind, intervals in by_kind.items() if kind != "nccl"},
                   nccl_kernels=dict(Counter(e["name"] for e in kernels if e["kind"] == "nccl")),
                   compute_by_phase_us={kind: duration(intervals) for kind, intervals in by_kind.items() if kind != "nccl"},
                   device_span_us=max(e["ts"] + e["dur"] for e in kernels) - min(e["ts"] for e in kernels),
                   operations={name: dict(kernel_count=len(selected), gpu_us=duration((e["ts"], e["ts"]+e["dur"]) for e in selected))
                               for name in sorted({e["operation"] for e in kernels})
                               if (selected := [e for e in kernels if e["operation"] == name])},
                   kernel_count=len(kernels))
    return dict(manifest=manifest, base=trace["baseTimeNanoseconds"], host=trace["host_name"],
                metrics=metrics, kernels=kernels, phases=phases, intervals=by_kind,
                memory=[e for e in events if e.get("name") == "[memory]"
                        and windows[0][0] <= e["ts"] <= windows[0][1]
                        and e.get("args", {}).get("Device Type") == 1],
                copies=copies,
                annotations=[e for e in annotations if windows[0][0] <= e["ts"] <= windows[0][1]
                             and e["name"].startswith(("pipeline/", "qwen/layer/", "training/step/", "training/optimizer"))])
