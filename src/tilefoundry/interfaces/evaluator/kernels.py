"""Tensor kernels shared by HIR and instruction-backed evaluation."""

from __future__ import annotations

from tilefoundry import utils
from tilefoundry.ir import expr


def gather(
    source: utils.torch.Tensor, index: utils.torch.Tensor, dim: int, fill: float | None
) -> utils.torch.Tensor:
    """Select slices along dim, filling out-of-range slices when fill is given."""
    if fill is None:
        return utils.torch.index_select(source, dim, index)
    valid = (index >= 0) & (index < source.shape[dim])
    shape = list(source.shape)
    shape[dim] = index.numel()
    data = utils.torch.full(shape, fill, dtype=source.dtype, device=source.device)
    selected = utils.torch.index_select(source, dim, index[valid])
    positions = utils.torch.nonzero(valid, as_tuple=True)[0]
    data.index_copy_(dim, positions, selected)
    return data


def binary(kind: expr.BinaryKind, lhs: utils.torch.Tensor, rhs: utils.torch.Tensor) -> utils.torch.Tensor:
    """Apply the shared HIR and scheduled binary operation."""
    operations = {
        expr.BinaryKind.ADD: utils.torch.add,
        expr.BinaryKind.SUB: utils.torch.sub,
        expr.BinaryKind.MUL: utils.torch.mul,
        expr.BinaryKind.DIV: utils.torch.div,
        expr.BinaryKind.FLOOR_DIV: utils.torch.floor_divide,
        expr.BinaryKind.MOD: utils.torch.remainder,
        expr.BinaryKind.MIN: utils.torch.minimum,
        expr.BinaryKind.MAX: utils.torch.maximum,
        expr.BinaryKind.EQ: utils.torch.eq,
        expr.BinaryKind.NE: utils.torch.ne,
        expr.BinaryKind.LT: utils.torch.lt,
        expr.BinaryKind.LE: utils.torch.le,
        expr.BinaryKind.GT: utils.torch.gt,
        expr.BinaryKind.GE: utils.torch.ge,
        expr.BinaryKind.AND: utils.torch.logical_and,
        expr.BinaryKind.OR: utils.torch.logical_or,
    }
    return operations[kind](lhs, rhs)


__all__ = ["binary", "gather"]
