"""Partition the installed Qwen reference while preserving its router auxiliary loss."""

import copy

import torch
from torch import nn
from transformers.masking_utils import create_causal_mask
from transformers.loss.loss_utils import ForCausalLMLoss
from transformers.models.qwen3_moe.modeling_qwen3_moe import load_balancing_loss_func


class QwenStage(nn.Module):
    """Own a contiguous layer interval and carry earlier router outputs to the loss."""

    def __init__(self, reference, index, stages):
        super().__init__()
        self.config = reference.config
        self.pack = None
        self.route_oracle = None
        self.microbatch = 0
        self.routing_comparisons = []
        self.first, self.last = index == 0, index == stages - 1
        width = self.config.num_hidden_layers // stages
        self.begin, self.end = index * width, (index + 1) * width
        self.model = nn.Module()
        if self.first:
            self.model.embed_tokens = copy.deepcopy(reference.model.embed_tokens)
        self.model.layers = nn.ModuleDict({str(i): copy.deepcopy(reference.model.layers[i])
                                          for i in range(self.begin, self.end)})
        self.model.rotary_emb = copy.deepcopy(reference.model.rotary_emb)
        if self.last:
            self.model.norm = copy.deepcopy(reference.model.norm)
            self.lm_head = copy.deepcopy(reference.lm_head)

    def forward(self, value, earlier_routers=None):
        hidden = self.model.embed_tokens(value) if self.first else value
        batch, tokens, _ = hidden.shape
        positions = torch.arange(tokens, device=hidden.device)
        if self.pack is None:
            position_ids = positions.unsqueeze(0)
            mask = create_causal_mask(config=self.config, inputs_embeds=hidden,
                                      attention_mask=None, cache_position=positions,
                                      past_key_values=None, position_ids=position_ids)
        else:
            position_ids = self.pack.local(self.pack.positions)
            mask = self.pack.local(self.pack.mask, 2)
        rotary = self.model.rotary_emb(hidden, position_ids=position_ids)
        routers = [] if earlier_routers is None else list(earlier_routers.unbind(dim=1))

        def capture(module, inputs, output):
            if self.route_oracle is not None:
                expected, probabilities = self.route_oracle[name][self.microbatch]
                actual = output[2].reshape(batch, tokens, -1)
                expected = self.pack.local(expected)
                probabilities = self.pack.local(probabilities)
                actual_probabilities = output[0].reshape(batch, tokens, -1)
                torch.testing.assert_close(actual_probabilities, probabilities, atol=1e-7, rtol=1e-4)
                mismatch = (actual.sort(dim=-1).values != expected.sort(dim=-1).values).any(-1)
                delta = (actual_probabilities - probabilities).abs().amax(-1)
                top = probabilities.topk(self.config.num_experts_per_tok + 1).values
                margin = top[..., -2] - top[..., -1]
                ambiguous = margin <= 2 * delta + 2 * torch.finfo(probabilities.dtype).eps * top[..., 0]
                if mismatch.any() and (self.routing_check == "strict" or (mismatch & ~ambiguous).any()):
                    raise AssertionError(f"layer {name} microbatch {self.microbatch}: "
                                         f"{mismatch.sum().item()} tokens changed selected expert set; "
                                         f"{(mismatch & ~ambiguous).sum().item()} exceed rounding bound")
                self.routing_comparisons.append(dict(layer=name, microbatch=self.microbatch,
                                                      changed_sets=mismatch.sum().item(),
                                                      max_probability_error=delta.max().item()))
                if self.routing_check == "replay-near-ties":
                    indices = expected.reshape(-1, self.config.num_experts_per_tok)
                    weights = output[0].gather(-1, indices)
                    weights = weights / weights.sum(-1, keepdim=True)
                    output = output[0], weights, indices
            routers.append(output[0].reshape(batch, tokens, self.config.num_experts))
            return output

        for name, layer in self.model.layers.items():
            handle = layer.mlp.gate.register_forward_hook(capture)
            try:
                with torch.profiler.record_function(f"qwen/layer/{name}"):
                    hidden = layer(hidden, attention_mask=mask, position_ids=position_ids,
                                   cache_position=positions, position_embeddings=rotary, use_cache=False)
            finally:
                handle.remove()
        output = self.lm_head(self.model.norm(hidden)) if self.last else hidden
        return output, torch.stack(routers, dim=1)


def pipeline_loss(outputs, labels, config, microbatches=1):
    """Match HF's per-microbatch CE and auxiliary loss across every decoder layer."""
    logits, routers = outputs
    router_tuple = tuple(value.reshape(-1, config.num_experts) for value in routers.unbind(dim=1))
    auxiliary = load_balancing_loss_func(router_tuple, config.num_experts, config.num_experts_per_tok)
    return (ForCausalLMLoss(logits, labels, config.vocab_size)
            + config.router_aux_loss_coef * auxiliary) / microbatches
