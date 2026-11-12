"""Stress zero-destination EP ranks and repeated token accumulation against HF."""

import os
from types import SimpleNamespace

import torch
import torch.distributed as dist
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeExperts

from config import TINY
from packed import Experts


def main():
    torch.set_num_threads(1)
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", device_id=torch.device("cuda", torch.cuda.current_device()))
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world == 8
    torch.manual_seed(1234)
    oracle = Qwen3MoeExperts(TINY.transformers_config()).cuda()
    with torch.no_grad():
        for parameter in oracle.parameters():
            parameter.normal_(std=0.02)
    parallel = SimpleNamespace(ep=world, ep_rank=rank, ep_group=dist.group.WORLD,
                               contract=lambda *args, **kwargs: None)
    local = Experts(oracle, parallel)
    for step in range(3):
        tokens = rank + 1 + step
        hidden = torch.randn(tokens, TINY.hidden, device="cuda", requires_grad=True)
        weights = torch.full((tokens, TINY.top_k), 1 / TINY.top_k, device="cuda", requires_grad=True)
        indices = torch.arange(TINY.top_k, device="cuda").expand(tokens, -1)
        expected = oracle(hidden, indices, weights)
        expected.square().sum().backward()
        hidden_grad, weights_grad = hidden.grad.clone(), weights.grad.clone()
        hidden.grad = weights.grad = None
        actual = local(hidden, indices, weights)
        actual.square().sum().backward()
        torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-5)
        torch.testing.assert_close(hidden.grad, hidden_grad, atol=1e-7, rtol=1e-5)
        torch.testing.assert_close(weights.grad, weights_grad, atol=1e-7, rtol=1e-5)
        for reference, parameter in zip(oracle.parameters(), local.parameters()):
            dist.all_reduce(reference.grad)
            want = reference.grad.chunk(world)[rank] / world
            torch.testing.assert_close(parameter.grad, want, atol=1e-7, rtol=1e-5)
        oracle.zero_grad(set_to_none=True)
        local.zero_grad(set_to_none=True)
    if rank == 0:
        print("passed: three uneven EP8 exchanges, four empty destinations, top-8 token accumulation and gradients")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
