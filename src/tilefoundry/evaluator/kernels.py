"""Tensor kernels shared by HIR and instruction-backed evaluation."""

from __future__ import annotations

import torch


def gather(
    source: torch.Tensor, index: torch.Tensor, dim: int, fill: float | None
) -> torch.Tensor:
    """Select slices along dim, filling out-of-range slices when fill is given."""
    if fill is None:
        return torch.index_select(source, dim, index)
    valid = (index >= 0) & (index < source.shape[dim])
    shape = list(source.shape)
    shape[dim] = index.numel()
    data = torch.full(shape, fill, dtype=source.dtype, device=source.device)
    selected = torch.index_select(source, dim, index[valid])
    positions = torch.nonzero(valid, as_tuple=True)[0]
    data.index_copy_(dim, positions, selected)
    return data
