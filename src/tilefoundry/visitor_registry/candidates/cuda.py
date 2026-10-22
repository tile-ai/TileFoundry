"""CUDA HIR-to-TIR candidate pairings."""

from dataclasses import replace
from math import prod

from tilefoundry.ir.hir.nn.matmul import MatMul
from tilefoundry.ir.hir.sharding.reshard import Reshard
from tilefoundry.ir.hir.tensor.index_select import IndexSelect
from tilefoundry.ir.tir.async_copy import CopyAsync
from tilefoundry.ir.tir.cuda.memory.copy_async_tensor import CopyAsyncTensor
from tilefoundry.ir.tir.cuda.memory.ldmatrix import LdMatrix
from tilefoundry.ir.tir.cuda.memory.stmatrix import StMatrix
from tilefoundry.ir.tir.cuda.nn.mma import TiledMma
from tilefoundry.ir.types import ComposedLayout, Layout, ShardLayout, Split
from tilefoundry.ir.types.int_tuple import flatten
from tilefoundry.ir.types.layout import resolve_layout
from tilefoundry.ir.types.shard_layout import layout_axis_to_tensor_axis

from . import register_candidates


def _matrix_type(type_, reverse):
    """Describe the same matrix bytes with reversed logical axes when requested."""
    if not reverse:
        return type_

    def permuted(layout):
        layout = resolve_layout(layout, type_.shape)
        if isinstance(layout, ComposedLayout):
            return replace(layout, outer=permuted(layout.outer))
        shard = layout if isinstance(layout, ShardLayout) else None
        local = shard.layout if shard is not None else layout
        axes = layout_axis_to_tensor_axis(
            tuple(prod(flatten(mode)) for mode in local.shape), type_.shape
        )
        order = tuple(
            index
            for axis in reversed(range(len(type_.shape)))
            for index, held in enumerate(axes)
            if held == axis
        )
        result = Layout(
            tuple(local.shape[index] for index in order),
            None if local.strides is None else tuple(local.strides[index] for index in order),
        )
        if shard is None:
            return result
        positions = {old: new for new, old in enumerate(order)}
        attrs = tuple(
            Split(positions[attr.axis]) if isinstance(attr, Split) else attr for attr in shard.attrs
        )
        return replace(shard, layout=result, attrs=attrs)

    return replace(type_, shape=tuple(reversed(type_.shape)), layout=permuted(type_.layout))


@register_candidates(MatMul, (TiledMma,))
def _matmul_operands(call, instruction):
    """Map HIR physical matrix axes to the instruction's fixed M,K and K,N axes."""
    op = call.target
    lhs = _matrix_type(call.args[0].type, op.a_layout == "KM")
    rhs = _matrix_type(call.args[1].type, op.b_layout == "NK")
    return {}, (call.type, lhs, rhs)


register_candidates(Reshard, (CopyAsync, CopyAsyncTensor, LdMatrix, StMatrix))


@register_candidates(IndexSelect, (CopyAsync,))
def _index_select_operands(call, instruction):
    """Supply the table, result and index; the HIR fill value belongs to the copy."""
    return {"fill": call.target.fill_value}, (call.args[0].type, call.type, call.args[1].type)
