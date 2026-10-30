"""Two explicit HIR training steps against independent dense autograd and SGD."""

from __future__ import annotations

import pytest
import torch

from examples.fsdp_training import training_program
from tilefoundry.analysis.check import check_program
from tilefoundry.evaluator import evaluate


@pytest.mark.parametrize("participants", (1, 2, 4))
@pytest.mark.parametrize("retain", (False, True), ids=("release", "retain"))
def test_fsdp_training_matches_dense_two_step_momentum_sgd(participants, retain):
    module = training_program(participants, retain=retain)
    _, function = check_program(module, module.entry_function())
    generator = torch.Generator().manual_seed(183)
    w1 = torch.nn.Parameter(torch.randn(8, 8, generator=generator) / 4)
    w2 = torch.nn.Parameter(torch.randn(8, 4, generator=generator) / 4)
    optimizer = torch.optim.SGD((w1, w2), lr=0.01, momentum=0.9)
    state = (w1.detach().clone(), w2.detach().clone(), torch.zeros_like(w1), torch.zeros_like(w2))
    for step in range(2):
        x = (torch.randn(8, 8, generator=generator) / 4 + step / 10).requires_grad_()
        target = torch.randn(8, 4, generator=generator) / 4
        optimizer.zero_grad()
        expected_y = torch.relu(x @ w1) @ w2
        loss = torch.nn.functional.mse_loss(expected_y, target)
        loss.backward()
        expected = (expected_y.detach(), loss.detach(), x.grad, w1.grad.clone(), w2.grad.clone())
        before = tuple(item.clone() for item in state)
        actual = evaluate(function, x.detach(), target, *state)
        for item, saved in zip(state, before, strict=True):
            assert torch.equal(item, saved)
        normalized = (*actual[:2], actual[2] / participants, *actual[3:5])
        for item, reference in zip(normalized, expected, strict=True):
            torch.testing.assert_close(item, reference, rtol=1e-5, atol=1e-6)
        optimizer.step()
        expected_state = (w1, w2, optimizer.state[w1]["momentum_buffer"], optimizer.state[w2]["momentum_buffer"])
        state = actual[5:]
        for item, reference in zip(state, expected_state, strict=True):
            torch.testing.assert_close(item, reference, rtol=1e-5, atol=1e-6)
