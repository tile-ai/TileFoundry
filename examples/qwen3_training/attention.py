"""Packed causal FA4 attention, independent of CP transport and optimizer policy."""

from itertools import accumulate

import torch

from tilefoundry import utils

flash = utils.LazyModule("flash_attn.cute")
DTYPE = torch.bfloat16


def segments(documents, padding, start, length):
    """Pair each local query segment with its document's causal KV prefix.

    Truncating KV at the last local query preserves bottom-right causal alignment
    even when a CP cut crosses a document. Padding rows attend only to themselves.
    """
    ends = tuple(accumulate(documents))
    begin, pieces = 0, []
    for end in ends:
        left, right = max(start, begin), min(start + length, end)
        if right > left:
            pieces.append((left - start, right - start, begin, right))
        begin = end
    for position in range(max(start, begin), min(start + length, begin + padding)):
        pieces.append((position - start, position - start + 1, position, position + 1))
    if sum(right - left for left, right, _, _ in pieces) != length:
        raise ValueError("packed attention segments do not cover the local query")
    return tuple(pieces)


def packed(query, key, value, pieces, *, scale=None):
    """B,H,S,D inputs; FP32 parameters with BF16 attention arithmetic."""
    if query.shape[0] != 1:
        raise ValueError("packed microbatches must have batch dimension one")
    q, k, v = [x[0].transpose(0, 1).contiguous().to(DTYPE) for x in (query, key, value)]
    keys = torch.cat([k[left:right] for _, _, left, right in pieces])
    values = torch.cat([v[left:right] for _, _, left, right in pieces])
    q_lengths = [right - left for left, right, _, _ in pieces]
    k_lengths = [right - left for _, _, left, right in pieces]
    cuq = torch.tensor((0, *accumulate(q_lengths)), device=q.device, dtype=torch.int32)
    cuk = torch.tensor((0, *accumulate(k_lengths)), device=q.device, dtype=torch.int32)
    with torch.profiler.record_function("qwen/attention"):
        result, _ = flash.flash_attn_varlen_func(
            q, keys, values, cu_seqlens_q=cuq, cu_seqlens_k=cuk,
            max_seqlen_q=max(q_lengths), max_seqlen_k=max(k_lengths),
            softmax_scale=scale, causal=True,
        )
    return result[None].to(query.dtype), None


def reference(module, query, key, value, attention_mask, scaling=None, **kwargs):
    pack = module.get_pack()
    pieces = segments(pack.summary["documents"], pack.summary["padding"], 0, query.shape[2])
    return packed(query, key, value, pieces, scale=scaling)
