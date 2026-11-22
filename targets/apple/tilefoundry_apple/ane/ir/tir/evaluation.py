"""Torch evaluation helpers shared by the ANE Ops: every Op writes its dst operand."""

from tilefoundry import utils


def f16(value):
    """Round to fp16 with ties away from zero, as each ANE arithmetic step does.

    torch rounds a tie to the even fp16 value; here a value exactly halfway between
    two fp16 values takes the one farther from zero. The result keeps *value*'s dtype.
    """
    torch = utils.torch
    magnitude = value.abs()
    nearest = magnitude.to(torch.float16)
    above = (nearest.view(torch.int16) + 1).view(torch.float16)
    below = nearest.to(magnitude.dtype)
    tie = (below < magnitude) & (magnitude - below == above.to(magnitude.dtype) - magnitude)
    return torch.copysign(torch.where(tie, above, nearest).to(value.dtype), value)


def write(ctx, result):
    """Copy *result* into the destination operand, narrowing to its dtype, and return it."""
    destination = ctx.args[0]
    destination.data.copy_(result.to(destination.data.dtype))
    return destination


__all__ = ["f16", "write"]
