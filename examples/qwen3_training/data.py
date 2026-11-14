"""Packed documents, causal targets and loss normalization for Qwen training."""

import torch
import torch.distributed as dist
from torch.nn import functional as F
from transformers.models.qwen3_moe.modeling_qwen3_moe import load_balancing_loss_func

from .parallel import gather


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
        self.segments = segments
        self.positions = positions[None]
        self.valid = (segments >= 0)[None]
        next_valid = torch.zeros_like(segments, dtype=torch.bool)
        next_valid[:-1] = (segments[:-1] == segments[1:]) & (segments[:-1] >= 0)
        self.next_valid = next_valid
        self.valid_targets = next_valid.sum()
        self.denominator = self.local(next_valid, 0).sum() * microbatches
        dist.all_reduce(self.denominator, group=dense_group)
        self.summary = dict(
            documents=[ends[0], ends[1] - ends[0], ends[2] - ends[1]],
            padding=padding,
            valid_targets=valid - len(ends),
            sequence=tokens,
        )

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
        aux = load_balancing_loss_func(
            router_tuple, config.num_experts, config.num_experts_per_tok, attention_mask=self.valid
        )
        ce = F.cross_entropy(
            logits.flatten(0, 1), targets.flatten(), ignore_index=-100, reduction="sum"
        )
        scale = self.parallel.data_size if reference else self.parallel.dense_size
        auxiliary = (
            config.router_aux_loss_coef * aux * self.valid_targets / (1 if reference else cp)
        )
        return (ce + auxiliary) * scale / self.denominator

    def reference(self, model, tokens):
        output = model.model(
            input_ids=tokens,
            attention_mask=None,
            position_ids=self.positions,
            output_router_logits=True,
            use_cache=False,
        )
        routers = torch.stack(
            [
                x.reshape(tokens.shape[0], tokens.shape[1], model.config.num_experts)
                for x in output.router_logits
            ],
            dim=1,
        )
        return self.loss(
            (model.lm_head(output.last_hidden_state), routers),
            self.targets(tokens),
            model.config,
            reference=True,
        )
