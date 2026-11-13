"""One rank map for pipeline, context, expert and optimizer domains."""

import copy

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.nn import functional as distributed
from torch.nn import functional as F
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from tilefoundry.ir import types
from tilefoundry.ir.hir.sharding import Collective, Exchange

from . import attention
from .trace import collective_function


class Parallel:
    """EP partitions the data axis; CP partitions each data lane's sequence."""

    def __init__(self, stages, cp=1, ep=1, hsdp_replicas=1):
        world, rank = dist.get_world_size(), dist.get_rank()
        if min(stages, cp, ep, hsdp_replicas) < 1 or world % stages:
            raise ValueError("parallelism dimensions must be positive and divide their rank domains")
        self.cp, self.ep = cp, ep
        self.dense_size = world // stages
        self.data_size = self.dense_size // cp
        if self.dense_size % (cp * ep) or (self.dense_size // ep) % hsdp_replicas:
            raise ValueError("parallelism dimensions must divide their rank domains")
        self.stage, local = divmod(rank, self.dense_size)
        self.data_rank, self.cp_rank = divmod(local, cp)
        self.outer_rank, self.ep_rank = divmod(self.data_rank, ep)
        outer = self.data_size // ep
        self.mesh = types.Mesh((types.Topology("gpu", world),),
                               types.Layout((stages, outer, ep, cp)), ("pp", "data", "ep", "cp"))
        self.local_rank, self.size = local, self.dense_size
        self.shards = self.size // hsdp_replicas
        self.replicate, self.shard = None, None
        self.expert_replicate, self.expert_shard = None, None
        memberships = [("pipeline", list(range(lane, world, self.size))) for lane in range(self.size)]

        def sharding_groups(prefix, domain):
            layout = torch.tensor(domain).reshape(hsdp_replicas, -1)
            if rank in domain:
                setattr(self, prefix+"ranks", layout)
            if hsdp_replicas > 1:
                memberships.extend((prefix+"shard", row.tolist()) for row in layout)
                memberships.extend((prefix+"replicate", column.tolist()) for column in layout.T)

        for stage in range(stages):
            base = stage * self.size
            dense = list(range(base, base + self.size))
            memberships.append(("data", dense))
            sharding_groups("", dense)
            memberships.extend(("cp_group", [base + data * cp + c for c in range(cp)])
                               for data in range(self.data_size))
            memberships.extend(("ep_group", [base + (data * ep + e) * cp + c for e in range(ep)])
                               for data in range(outer) for c in range(cp))
            for e in range(ep if ep > 1 else 0):
                experts = [base + (data * ep + e) * cp + c for data in range(outer) for c in range(cp)]
                memberships.append(("expert_group", experts))
                sharding_groups("expert_", experts)
        for name, ranks in memberships:
            group = dist.new_group(ranks)
            if rank in ranks:
                setattr(self, name, group)
        if ep == 1:
            self.expert_group, self.expert_ranks = self.data, self.ranks
            self.expert_shard, self.expert_replicate = self.shard, self.replicate
        self.contracts = {}
        self.record = True

    def contract(self, kind, group, shape, dtype, *, axis=0, counts=None):
        """Typecheck observed calls without claiming dynamic-count HIR semantics."""
        if not self.record or dist.get_world_size(group) == 1:
            return
        ranks = tuple(dist.get_process_group_ranks(group))
        key = kind, ranks, tuple(shape), str(dtype), axis, counts
        if key in self.contracts:
            self.contracts[key]["calls"] += 1
            return
        mesh = (self.mesh[self.stage, self.outer_rank, :, self.cp_rank] if group is self.ep_group
                else self.mesh[self.stage, self.outer_rank, self.ep_rank, :])
        context = f"nccl/{group.group_name}"
        if counts is None:
            op = Collective(participants=mesh, context=context, kind=kind, axis=axis,
                            reduction="sum" if kind == "reduce_scatter" else None)
        else:
            op = Exchange(participants=mesh, context=context, counts=counts)
        function = collective_function(op, shape, dtype)
        self.contracts[key] = dict(kind=kind, participants=ranks, context=context,
                                   input_shape=shape, output_shape=function.return_type.shape,
                                   dtype=str(dtype), counts=counts, axis=axis, calls=1)

    def apply(self, stage):
        ALL_ATTENTION_FUNCTIONS.register("qwen3_context", context_attention)
        for layer in stage.model.layers.values():
            layer.self_attn.config = copy.copy(layer.self_attn.config)
            layer.self_attn.config._attn_implementation = "qwen3_context"
            layer.self_attn.parallel = self
            layer.self_attn.get_pack = lambda: stage.pack
            if self.ep > 1:
                layer.mlp.experts = Experts(layer.mlp.experts, self)


def gather(value, group, axis, parallel):
    if dist.get_world_size(group) == 1:
        return value
    parallel.contract("all_gather", group, tuple(value.shape), value.dtype, axis=axis)
    result = torch.cat(distributed.all_gather(value.contiguous(), group=group), dim=axis)
    parallel.contract("reduce_scatter", group, tuple(result.shape), value.dtype, axis=axis)
    return result


def context_attention(module, query, key, value, attention_mask, **kwargs):
    parallel = module.parallel
    with torch.profiler.record_function("communication/cp_kv"):
        key, value = gather(torch.stack((key, value)), parallel.cp_group, 3, parallel).unbind(0)
    pack = module.get_pack()
    pieces = attention.segments(pack.summary["documents"],pack.summary["padding"],pack.start,pack.length)
    return attention.packed(query,key,value,pieces,scale=kwargs.get("scaling"))


class ScaleGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, factor):
        ctx.factor = factor
        return value.view_as(value)

    @staticmethod
    def backward(ctx, gradient):
        return gradient * ctx.factor, None


class Experts(nn.Module):
    """Retain local experts and route token rows with reversible uneven all-to-all."""

    def __init__(self, original, parallel):
        super().__init__()
        self.parallel = parallel
        self.num_experts = original.num_experts // parallel.ep
        start = parallel.ep_rank * self.num_experts
        self.gate_up_proj = nn.Parameter(original.gate_up_proj[start:start + self.num_experts].detach().clone())
        self.down_proj = nn.Parameter(original.down_proj[start:start + self.num_experts].detach().clone())
        self.act_fn = original.act_fn

    def forward(self, hidden, indices, weights):
        parallel, width = self.parallel, hidden.shape[-1]
        order = indices.flatten().argsort(stable=True)
        experts = indices.flatten()[order]
        token_ids = torch.arange(hidden.shape[0], device=hidden.device).repeat_interleave(indices.shape[1])[order]
        send = torch.bincount(experts // self.num_experts, minlength=parallel.ep)
        rows = [torch.empty_like(send) for _ in range(parallel.ep)]
        with torch.profiler.record_function("communication/ep_counts"):
            dist.all_gather(rows, send, group=parallel.ep_group)
        matrix = torch.stack(rows).cpu().tolist()
        send_counts = matrix[parallel.ep_rank]
        recv_counts = [row[parallel.ep_rank] for row in matrix]
        parallel.contract("all_gather", parallel.ep_group, tuple(send.shape), send.dtype)
        counts = tuple(tuple(n * width for n in row) for row in matrix)
        metadata_counts = tuple(tuple(row) for row in matrix)
        parallel.contract("exchange", parallel.ep_group, (max(map(sum, metadata_counts)),), indices.dtype,
                          counts=metadata_counts)
        parallel.contract("exchange", parallel.ep_group, (max(map(sum, counts)),), hidden.dtype, counts=counts)
        reverse = tuple(zip(*counts))
        parallel.contract("exchange", parallel.ep_group, (max(map(sum, reverse)),), hidden.dtype, counts=reverse)
        incoming_ids = torch.empty(sum(recv_counts), dtype=indices.dtype, device=hidden.device)
        with torch.profiler.record_function("communication/ep_dispatch"):
            dist.all_to_all_single(incoming_ids, experts, recv_counts, send_counts, group=parallel.ep_group)
            received = distributed.all_to_all_single(hidden.new_empty((sum(recv_counts), width)),
                                                      hidden[token_ids], recv_counts, send_counts,
                                                      group=parallel.ep_group)
        local_ids = incoming_ids % self.num_experts
        local_order = local_ids.argsort(stable=True)
        lengths = torch.bincount(local_ids, minlength=self.num_experts).cpu().tolist()
        gate_up = ScaleGradient.apply(self.gate_up_proj, 1 / parallel.ep)
        down = ScaleGradient.apply(self.down_proj, 1 / parallel.ep)
        chunks, start = [], 0
        for expert, length in enumerate(lengths):
            if length:
                selected = received[local_order[start:start + length]]
                gate, up = F.linear(selected, gate_up[expert]).chunk(2, dim=-1)
                chunks.append(F.linear(self.act_fn(gate) * up, down[expert]))
                start += length
        if chunks:
            computed = torch.cat(chunks)[local_order.argsort()]
        else:
            computed = received + (gate_up.flatten()[:1].sum() + down.flatten()[:1].sum()) * 0
        with torch.profiler.record_function("communication/ep_return"):
            returned = distributed.all_to_all_single(hidden.new_empty((sum(send_counts), width)),
                                                      computed, send_counts, recv_counts, group=parallel.ep_group)
        contributions = (returned * weights.flatten()[order, None])[order.argsort()]
        contributions = contributions.reshape(*indices.shape, width)
        expert_order = indices.argsort(dim=1)
        contributions = contributions.gather(1, expert_order[:, :, None].expand_as(contributions))
        result = torch.zeros_like(hidden)
        for contribution in contributions.unbind(1):
            result = result + contribution
        return result
