"""Torch evaluation helpers shared by the Metal Ops: every Op writes its dst operand."""

from tilefoundry import utils


def write(ctx, result):
    """Copy *result* into the destination operand, narrowing to its dtype, and return it."""
    destination = ctx.args[0]
    destination.data.copy_(result.to(destination.data.dtype))
    return destination


def widened(value):
    """Lane data torch can combine: u16 and u32 as int64; write() narrows back with wrap-around."""
    data = value.data
    return data.to(utils.torch.int64) if data.dtype in (utils.torch.uint16, utils.torch.uint32) else data


__all__ = ["widened", "write"]
