"""Packed-document CP attention and variable-count EP using PyTorch collectives."""

import copy

import torch
import torch.distributed as dist
from model import collective_function
from torch import nn
from torch.distributed.nn import functional as distributed
from torch.nn import functional as F
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen3_moe.modeling_qwen3_moe import load_balancing_loss_func

from tilefoundry.ir import types
from tilefoundry.ir.hir.sharding import Collective, Exchange


class Parallel:
    """EP partitions the data axis; CP partitions each data lane's sequence."""

    def __init__(self, stages, cp, ep):
        world, rank = dist.get_world_size(), dist.get_rank()
        self.cp, self.ep = cp, ep
        self.dense_size = world // stages
        self.data_size = self.dense_size // cp
        self.stage, local = divmod(rank, self.dense_size)
        self.data_rank, self.cp_rank = divmod(local, cp)
        self.outer_rank, self.ep_rank = divmod(self.data_rank, ep)
        outer = self.data_size // ep
        self.mesh = types.Mesh((types.Topology("gpu", world),),
                               types.Layout((stages, outer, ep, cp)), ("pp", "data", "ep", "cp"))
        for stage in range(stages):
            for data in range(self.data_size):
                ranks = [stage * self.dense_size + data * cp + c for c in range(cp)]
                group = dist.new_group(ranks)
                if rank in ranks:
                    self.cp_group = group
            for data in range(outer):
                for c in range(cp):
                    ranks = [stage * self.dense_size + (data * ep + e) * cp + c for e in range(ep)]
                    group = dist.new_group(ranks)
                    if rank in ranks:
                        self.ep_group = group
            for e in range(ep):
                ranks = [stage * self.dense_size + (data * ep + e) * cp + c
                         for data in range(outer) for c in range(cp)]
                group = dist.new_group(ranks)
                if rank in ranks:
                    self.expert_group = group
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
    return sdpa_attention_forward(module, query, key, value, attention_mask, **kwargs)


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


class Pack:
    """Document boundaries, reset positions, cross-CP targets and a global denominator."""

    def __init__(self, parallel, tokens, microbatches, step, device, dense_group):
        self.parallel, self.microbatches = parallel, microbatches
        self.length = tokens // parallel.cp
        self.start = parallel.cp_rank * self.length
        padding = 1 + (7 * parallel.data_rank + step) % max(2, tokens // 16)
        valid = tokens - padding
        ends = (tokens * 5 // 8, valid * 7 // 8, valid)
        segments = torch.full((tokens,), -1, device=device, dtype=torch.long)
        positions = torch.zeros_like(segments)
        start = 0
        for segment, end in enumerate(ends):
            segments[start:end] = segment
            positions[start:end] = torch.arange(end - start, device=device)
            start = end
        offset = torch.arange(tokens, device=device)
        allowed = (segments[:, None] == segments[None, :]) & (offset[:, None] >= offset[None, :])
        allowed &= segments[:, None] >= 0
        allowed |= (segments[:, None] < 0) & (offset[:, None] == offset[None, :])
        self.mask = allowed[None, None]
        self.positions = positions[None]
        self.valid = (segments >= 0)[None]
        next_valid = torch.zeros_like(segments, dtype=torch.bool)
        next_valid[:-1] = (segments[:-1] == segments[1:]) & (segments[:-1] >= 0)
        self.next_valid = next_valid
        self.valid_targets = next_valid.sum()
        self.denominator = self.local(next_valid, 0).sum() * microbatches
        dist.all_reduce(self.denominator, group=dense_group)
        self.summary = dict(documents=[ends[0], ends[1] - ends[0], ends[2] - ends[1]],
                            padding=padding, valid_targets=valid - len(ends), sequence=tokens)

    def local(self, tensor, axis=1):
        return tensor.narrow(axis, self.start, self.length).contiguous()

    def targets(self, tokens):
        labels = tokens.roll(-1, dims=1)
        return labels.masked_fill(~self.next_valid[None], -100)

    def loss(self, outputs, targets, config, *, reference=False):
        logits, routers = outputs
        cp = self.parallel.cp
        if not reference:
            routers = gather(routers, self.parallel.cp_group, 2, self.parallel)
        router_tuple = tuple(value.reshape(-1, config.num_experts) for value in routers.unbind(1))
        aux = load_balancing_loss_func(router_tuple, config.num_experts, config.num_experts_per_tok,
                                      attention_mask=self.valid)
        ce = F.cross_entropy(logits.flatten(0, 1), targets.flatten(), ignore_index=-100, reduction="sum")
        scale = self.parallel.data_size if reference else self.parallel.dense_size
        auxiliary = config.router_aux_loss_coef * aux * self.valid_targets / (1 if reference else cp)
        return (ce + auxiliary) * scale / self.denominator

    def reference(self, model, tokens):
        output = model.model(input_ids=tokens, attention_mask=self.mask, position_ids=self.positions,
                             output_router_logits=True, use_cache=False)
        routers = torch.stack([x.reshape(tokens.shape[0], tokens.shape[1], model.config.num_experts)
                               for x in output.router_logits], dim=1)
        return self.loss((model.lm_head(output.last_hidden_state), routers), self.targets(tokens),
                         model.config, reference=True)
