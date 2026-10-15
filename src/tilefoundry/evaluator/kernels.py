"""Tensor kernels shared by HIR and instruction-backed evaluation."""

from __future__ import annotations

import torch

from tilefoundry.ir.core.kinds import BinaryKind


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


def binary(kind: BinaryKind, lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """Apply the shared HIR and scheduled binary operation."""
    operations = {
        BinaryKind.ADD: torch.add,
        BinaryKind.SUB: torch.sub,
        BinaryKind.MUL: torch.mul,
        BinaryKind.DIV: torch.div,
        BinaryKind.FLOOR_DIV: torch.floor_divide,
        BinaryKind.MOD: torch.remainder,
        BinaryKind.MIN: torch.minimum,
        BinaryKind.MAX: torch.maximum,
        BinaryKind.EQ: torch.eq,
        BinaryKind.NE: torch.ne,
        BinaryKind.LT: torch.lt,
        BinaryKind.LE: torch.le,
        BinaryKind.GT: torch.gt,
        BinaryKind.GE: torch.ge,
        BinaryKind.AND: torch.logical_and,
        BinaryKind.OR: torch.logical_or,
    }
    return operations[kind](lhs, rhs)
