"""Target-neutral HIR-to-TIR candidate pairings."""

from dataclasses import replace
from itertools import zip_longest

from tilefoundry.interfaces import access, schedule, typeinfer
from tilefoundry.ir import expr, tir, types
from tilefoundry.ir.hir import math, nn, sharding, tensor
from tilefoundry.ir.tir import memory
from tilefoundry.ir.tir import nn as tir_nn

__all__ = ["combines_partial", "only_moves"]


def combines_partial(call) -> bool:
    """Whether a reshard resolves a Partial, which combines the units' parts."""
    return resolved_reduction(call.args[0].type, call.type) is not None


def only_moves(call) -> bool:
    """Whether a reshard only moves data: it resolves no Partial."""
    return not combines_partial(call)


@schedule.register_candidates(sharding.Reshard, (memory.Copy,), when=only_moves)
def reshard_operands(call, instruction):
    """Reshard states both transfer types; its landing attributes are already in the result.

    A reduction over no tensor axis combines the Partial the reshard resolves.
    """
    source, result = call.args[0].type, call.type
    if instruction is not tir.reduce.Reduce:
        return {}, (source, result)
    kind = expr.ReduceKind(resolved_reduction(source, result))
    return {"axes": (), "keepdim": True, "kind": kind}, (source, result)


schedule.register_candidates(sharding.Reshard, (tir.reduce.Reduce,), when=combines_partial)


def resolved_reduction(source, result) -> str | None:
    """The reduction a reshard performs by dropping its source's Partial axes, if any."""
    resolved = {
        kind
        for kind, kept in zip_longest(
            types.partial_reductions_by_axis(source.layout), types.partial_reductions_by_axis(result.layout)
        )
        if kind is not None and kept is None
    }
    if len(resolved) > 1:
        raise schedule.RefusalError(f"a reshard cannot combine Partial reductions {sorted(resolved)} at once")
    return resolved.pop() if resolved else None


schedule.register_candidates(math.Binary, (tir.arith.Binary,))
schedule.register_candidates(math.Unary, (tir.arith.Unary,))
schedule.register_candidates(math.Clamp, (tir.clamp.Clamp,))
schedule.register_candidates(nn.ReLU, (tir_nn.ReLU,))
schedule.register_candidates(tensor.Cast, (tir.cast.Cast,))
schedule.register_candidates(tensor.Reduce, (tir.reduce.Reduce,))
schedule.register_candidates(tensor.Where, (tir.where.Where,))


class OperandTypes:
    """The types of a call's operands, as the access relations read them."""

    def type_of(self, value):
        """The type *value* carries."""
        return value.type


@schedule.register_candidates(tensor.Transpose, (memory.Copy,))
def transpose_operands(call, instruction):
    """Read the permuted source view; Copy materializes its compact destination."""
    source = call.args[0].type

    layout = typeinfer.derive_output_shard_layout(
        (source,), access.relations_of(call, OperandTypes()), tuple(call.type.shape), fresh_strides=False
    )
    if layout is None:
        plain = source.layout.layout if isinstance(source.layout, types.ShardLayout) else source.layout

        def permuted(held):
            return types.Layout(
                tuple(held.shape[p] for p in call.target.perm),
                tuple(held.strides[p] for p in call.target.perm),
            )

        layout = tensor.derive_view_layout(replace(source, layout=plain), tuple(call.type.shape), permuted)
    if layout is None:
        raise ValueError("cannot transpose source layout")
    return {}, (replace(call.type, layout=layout), call.type)


@schedule.register_candidates(tensor.Arange, (tir.arange.Arange,))
def arange_operands(call, instruction):
    if not isinstance(call.target.start, int):
        raise schedule.RefusalError(f"Arange start must be a static integer, got {call.target.start!r}")
    return {"start": call.target.start, "step": call.target.step}, (call.type,)
