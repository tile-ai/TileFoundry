"""Independent dense PyTorch reference: autograd, SDPA, and momentum SGD."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .model import B, C, D, EPS, G, H, N, PARAMETER_NAMES, PARAMETER_SHAPES, T, V


def rotary(x):
    """Rotate adjacent pairs; HIR instead multiplies fixed rotation matrices."""
    frequencies = 10000.0 ** (-torch.arange(0, D, 2).float() / D)
    angles = torch.arange(T).float()[:, None] * frequencies
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even * angles.cos() - odd * angles.sin(),
                        even * angles.sin() + odd * angles.cos()), dim=-1).flatten(-2)


def fixed_inputs(tokens, targets):
    """Encode token lookup and fixed RoPE using existing HIR arithmetic."""
    frequencies = 10000.0 ** (-torch.arange(0, D, 2).float() / D)
    angles = torch.arange(T).float()[:, None] * frequencies
    rotation = torch.zeros(T, D, D)
    for pair in range(D // 2):
        i = 2 * pair
        rotation[:, i, i] = angles[:, pair].cos()
        rotation[:, i, i + 1] = angles[:, pair].sin()
        rotation[:, i + 1, i] = -angles[:, pair].sin()
        rotation[:, i + 1, i + 1] = angles[:, pair].cos()
    mask = torch.zeros(T, T).masked_fill(torch.ones(T, T).triu(1).bool(), -float("inf"))
    return (F.one_hot(tokens.reshape(N), V).float(), F.one_hot(targets.reshape(N), V).float(),
            rotation.reshape(1, 1, 1, T, D, D), mask.reshape(1, 1, 1, T, T))


class Reference(nn.Module):
    """Two bias-free blocks, affine-free RMSNorm, untied embedding and head."""

    def __init__(self, seed=183):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.weights = nn.ParameterList([
            nn.Parameter(torch.randn(shape, generator=generator) * 0.1)
            for shape in PARAMETER_SHAPES
        ])

    def forward(self, tokens):
        x = F.embedding(tokens, self.weights[0])
        for start in (1, 7):
            wq, wk, wv, wo, wu, wd = self.weights[start:start + 6]
            a = F.rms_norm(x, (C,), eps=EPS)
            q = F.linear(a, wq).reshape(B, T, H, D).transpose(1, 2)
            k = F.linear(a, wk).reshape(B, T, G, D).transpose(1, 2)
            v = F.linear(a, wv).reshape(B, T, G, D).transpose(1, 2)
            q = rotary(F.rms_norm(q, (D,), eps=EPS))
            k = rotary(F.rms_norm(k, (D,), eps=EPS))
            attended = F.scaled_dot_product_attention(
                q, k.repeat_interleave(H // G, dim=1), v.repeat_interleave(H // G, dim=1),
                is_causal=True,
            ).transpose(1, 2).reshape(B, T, C)
            x = x + F.linear(attended, wo)
            r = F.rms_norm(x, (C,), eps=EPS)
            x = x + F.linear(F.relu(F.linear(r, wu)).square(), wd)
        return F.linear(F.rms_norm(x, (C,), eps=EPS), self.weights[-1])


def check_training(function, *, steps=2, seed=183):
    """Check every gradient and state leaf over changing batches and two updates."""
    from tilefoundry.evaluator import evaluate

    model = Reference(seed)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    weights = tuple(p.detach().clone() for p in model.weights)
    momentum = tuple(torch.zeros_like(p) for p in weights)
    generator = torch.Generator().manual_seed(seed + 1)
    errors = {"logits": 0.0, "loss": 0.0, "gradients": 0.0, "parameters": 0.0, "momentum": 0.0}
    for _ in range(steps):
        sequence = torch.randint(V, (B, T + 1), generator=generator)
        tokens, targets = sequence[:, :-1], sequence[:, 1:]
        optimizer.zero_grad()
        logits = model(tokens)
        loss = F.cross_entropy(logits.reshape(N, V), targets.reshape(N))
        loss.backward()
        before = tuple(p.clone() for p in (*weights, *momentum))
        actual = evaluate(function, *fixed_inputs(tokens, targets), *weights, *momentum)
        for old, current in zip(before, (*weights, *momentum), strict=True):
            assert torch.equal(old, current), "HIR evaluation mutated its inputs"
        expected_gradients = tuple(p.grad.clone() for p in model.weights)
        # All parameters participate from step one: random nonzero projections
        # avoid a zero-initialized output projection hiding a backward mistake.
        assert all(g.abs().max() > 1e-8 for g in expected_gradients)
        optimizer.step()
        expected = ((logits.detach().reshape(N, V),), (loss.detach(),), expected_gradients,
                    tuple(model.weights), tuple(optimizer.state[p]["momentum_buffer"] for p in model.weights))
        observed = ((actual[0],), (actual[1],), *actual[2:])
        for (kind, previous), observed_group, expected_group in zip(errors.items(), observed, expected, strict=True):
            for index, (got, want) in enumerate(zip(observed_group, expected_group, strict=True)):
                torch.testing.assert_close(got, want, rtol=5e-4, atol=2e-5,
                                           msg=lambda message: f"{kind} {PARAMETER_NAMES[index]}: {message}")
                errors[kind] = max(errors[kind], float((got - want).abs().max().detach()))
        weights, momentum = actual[3:]
    return errors
