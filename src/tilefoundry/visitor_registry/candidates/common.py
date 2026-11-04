"""Target-neutral HIR-to-TIR candidate pairings."""

from dataclasses import replace

from tilefoundry.ir.core.kinds import ReduceKind
from tilefoundry.ir.hir.math.binary import Binary as HirBinary
from tilefoundry.ir.hir.math.clamp import Clamp as HirClamp
from tilefoundry.ir.hir.math.unary import Unary as HirUnary
from tilefoundry.ir.hir.nn.relu import ReLU as HirReLU
from tilefoundry.ir.hir.sharding.reshard import Reshard
from tilefoundry.ir.hir.tensor._view_layout import derive_view_layout
from tilefoundry.ir.hir.tensor.arange import Arange as HirArange
from tilefoundry.ir.hir.tensor.cast import Cast as HirCast
from tilefoundry.ir.hir.tensor.reduce import Reduce as HirReduce
from tilefoundry.ir.hir.tensor.transpose import Transpose
from tilefoundry.ir.hir.tensor.where import Where as HirWhere
from tilefoundry.ir.tir.arange import Arange as TirArange
from tilefoundry.ir.tir.arith import Binary as TirBinary
from tilefoundry.ir.tir.arith import Unary as TirUnary
from tilefoundry.ir.tir.cast import Cast as TirCast
from tilefoundry.ir.tir.clamp import Clamp as TirClamp
from tilefoundry.ir.tir.memory.copy import Copy
from tilefoundry.ir.tir.nn.relu import ReLU as TirReLU
from tilefoundry.ir.tir.reduce import Reduce as TirReduce
from tilefoundry.ir.tir.where import Where as TirWhere
from tilefoundry.ir.types import Layout
from tilefoundry.ir.types.shard_layout import ShardLayout
from tilefoundry.visitor_registry.access_relation import relations_of
from tilefoundry.visitor_registry.shard_propagate import (
    derive_output_shard_layout,
    partial_reductions_by_axis,
)

from ..instruction import ScheduleRefusal
from . import register_candidates


def combines_partial(call) -> bool:
    """Whether a reshard resolves a Partial, which combines the units' parts."""
    return _resolved_reduction(call.args[0].type, call.type) is not None


def only_moves(call) -> bool:
    """Whether a reshard only moves data: it resolves no Partial."""
    return not combines_partial(call)


@register_candidates(Reshard, (Copy,), when=only_moves)
def _reshard_operands(call, instruction):
    """Reshard states both transfer types; its landing attributes are already in the result.

    A reduction over no tensor axis combines the Partial the reshard resolves.
    """
    source, result = call.args[0].type, call.type
    if instruction is not TirReduce:
        return {}, (source, result)
    kind = ReduceKind(_resolved_reduction(source, result))
    return {"axes": (), "keepdim": True, "kind": kind}, (source, result)


register_candidates(Reshard, (TirReduce,), when=combines_partial)


def _resolved_reduction(source, result) -> str | None:
    """The reduction a reshard performs by dropping its source's Partial axes, if any."""
    before = partial_reductions_by_axis(source.layout)
    after = partial_reductions_by_axis(result.layout)
    after = after + (None,) * (len(before) - len(after))
    resolved = {kind for kind, kept in zip(before, after) if kind is not None and kept is None}
    if len(resolved) > 1:
        raise ScheduleRefusal(f"a reshard cannot combine Partial reductions {sorted(resolved)} at once")
    return resolved.pop() if resolved else None


register_candidates(HirBinary, (TirBinary,))
register_candidates(HirUnary, (TirUnary,))
register_candidates(HirClamp, (TirClamp,))
register_candidates(HirReLU, (TirReLU,))
register_candidates(HirCast, (TirCast,))
register_candidates(HirReduce, (TirReduce,))
register_candidates(HirWhere, (TirWhere,))


@register_candidates(Transpose, (Copy,))
def _transpose_operands(call, instruction):
    """Read the permuted source view; Copy materializes its compact destination."""
    source = call.args[0].type

    class Types:
        @staticmethod
        def type_of(value):
            return value.type

    layout = derive_output_shard_layout(
        (source,), relations_of(call, Types()), tuple(call.type.shape), fresh_strides=False
    )
    if layout is None:
        plain = source.layout.layout if isinstance(source.layout, ShardLayout) else source.layout

        def permuted(held):
            return Layout(
                tuple(held.shape[p] for p in call.target.perm),
                tuple(held.strides[p] for p in call.target.perm),
            )

        layout = derive_view_layout(replace(source, layout=plain), tuple(call.type.shape), permuted)
    if layout is None:
        raise ValueError("cannot transpose source layout")
    return {}, (replace(call.type, layout=layout), call.type)


@register_candidates(HirArange, (TirArange,))
def _arange_operands(call, instruction):
    if not isinstance(call.target.start, int):
        raise ScheduleRefusal(f"Arange start must be a static integer, got {call.target.start!r}")
    return {"start": call.target.start, "step": call.target.step}, (call.type,)
