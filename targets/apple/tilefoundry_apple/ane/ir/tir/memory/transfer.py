"""The rules a transfer between unified memory and the L2 buffer shares."""

from tilefoundry.interfaces import access, evaluator, typeinfer
from tilefoundry.interfaces import verify as interfaces_verify
from tilefoundry.ir.tir import memory
from tilefoundry_apple.ane.ir.tir import evaluation


def relations(call, ctx):
    """Walk the feature map of the transfer; a matrix operand is read through its linear view.

    The call produces its destination, so that relation comes twice.
    """
    shapes = tuple(tuple(ctx.type_of(arg).shape) for arg in call.args)
    domain = max(shapes, key=len)
    reads = tuple(
        access.identity(len(domain)) if shape == domain else access.linearized_view(domain, shape)
        for shape in shapes
    )
    return access.iterating(domain, (*reads, reads[1]))


def transfer_value(ctx):
    """Write the source, read in row-major order, as the shape of the destination."""
    src, dst = ctx.args
    return evaluation.write(dst, src.data.reshape(dst.data.shape))


def register_transfer(op):
    """Reuse Copy's type and verify rules; the value is the source in the destination's shape."""
    typeinfer.register(op)(typeinfer.registry.require(memory.Copy))
    access.register_relation(op)(relations)
    rule = interfaces_verify.registry.get(memory.Copy, None)
    if rule is not None:
        interfaces_verify.register(op)(rule)
    evaluator.register(op)(transfer_value)
    return op


__all__ = ["register_transfer"]
