"""Apply dmuon's public API to disjoint dense and expert parameter domains."""

import torch
import torch.distributed as dist
from torch import nn

import dedicated


def views(stage):
    dense, expert = nn.Module(), nn.Module()
    dense.model, expert.model = nn.Module(), nn.Module()
    boundaries = [set(), set()]
    if stage.first:
        dense.model.embed_tokens = stage.model.embed_tokens
        boundaries[0].add(stage.model.embed_tokens)
    dense.model.layers, expert.model.layers = nn.ModuleDict(), nn.ModuleDict()
    for name, layer in stage.model.layers.items():
        d, e = nn.Module(), nn.Module()
        d.input_layernorm = layer.input_layernorm
        d.self_attn = layer.self_attn
        d.post_attention_layernorm = layer.post_attention_layernorm
        d.mlp, e.mlp = nn.Module(), nn.Module()
        d.mlp.gate, e.mlp.experts = layer.mlp.gate, layer.mlp.experts
        dense.model.layers[name], expert.model.layers[name] = d, e
        boundaries[0].update((d.input_layernorm, d.self_attn, d.post_attention_layernorm, d.mlp.gate))
        boundaries[1].add(e.mlp.experts)
    if stage.last:
        dense.model.norm, dense.lm_head = stage.model.norm, stage.lm_head
        boundaries[0].update((stage.model.norm, stage.lm_head))
    return (dense, expert), boundaries


class ParallelOptimizer:
    def __init__(self, stage, parallel, dense_group, shape, *, overlap, ns_backend, forward_unshard):
        self.models, boundaries = views(stage)
        self.names = [tuple(dict(model.named_parameters())) for model in self.models]
        self.groups = dense_group, parallel.expert_group
        self.parallel, self.dense_group = parallel, dense_group
        count = shape.experts // parallel.ep
        self.expert_slice = slice(parallel.ep_rank * count, (parallel.ep_rank + 1) * count)
        self.optimizers = [dedicated.configure(model, group, shape, overlap=overlap, ns_backend=ns_backend,
                                               forward_unshard=forward_unshard, boundaries=boundary)
                           for model, group, boundary in zip(self.models, self.groups, boundaries)]

    def zero_grad(self, **kwargs):
        for optimizer in self.optimizers:
            optimizer.zero_grad(**kwargs)

    def step(self):
        for optimizer in self.optimizers:
            optimizer.step()

    def check_gradients(self, reference):
        results = [dedicated.check_gradients(model, reference, names, self.dense_group,
                                             expert_slice=self.expert_slice if i else None)
                   for i, (model, names) in enumerate(zip(self.models, self.names))]
        return sum(n for n, _ in results), max(error for _, error in results)

    def snapshot(self):
        result = dedicated.snapshot(self.models[0], self.groups[0])
        for name, weight in dedicated.snapshot(self.models[1], self.groups[1]).items():
            parts = [torch.empty_like(weight) for _ in range(self.parallel.ep)]
            dist.all_gather(parts, weight.contiguous(), group=self.parallel.ep_group)
            result[name] = torch.cat(parts, dim=0)
        return result
