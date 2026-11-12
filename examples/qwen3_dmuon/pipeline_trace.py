"""Check existing PyTorch pipeline transfers against TileFoundry Exchange contracts."""

import torch
import torch.distributed as dist
from torch.distributed.pipelining import PipelineStage

from tilefoundry import core
from tilefoundry.interfaces import evaluator
from tilefoundry.ir import types
from tilefoundry.ir.hir import sharding

from model import collective_function


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
            ("fwd", self._downstream_group), ("bwd", self._upstream_group))}
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
