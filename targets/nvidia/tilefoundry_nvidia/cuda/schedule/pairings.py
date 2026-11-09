"""CUDA HIR-to-TIR candidate pairings."""

from dataclasses import replace
from math import prod

from tilefoundry import schedule
from tilefoundry.interfaces import schedule as interfaces_schedule
from tilefoundry.ir import tir, types
from tilefoundry.ir.hir import nn, sharding, tensor
from tilefoundry.ir.types import int_tuple
from tilefoundry_nvidia.cuda.ir.tir import memory
from tilefoundry_nvidia.cuda.ir.tir import nn as tir_nn


def matrix_type(type_, reverse):
    """Describe the same matrix bytes with reversed logical axes when requested."""
    if not reverse:
        return type_

    def permuted(layout):
        if isinstance(layout, types.ComposedLayout):
            return replace(layout, outer=permuted(layout.outer))
        shard = layout if isinstance(layout, types.ShardLayout) else None
        local = shard.layout if shard is not None else layout
        axes = int_tuple.mode_axes(int_tuple.compatible_profile(tuple(prod(types.flatten(mode)) for mode in local.shape), type_.shape))
        order = tuple(
            index
            for axis in reversed(range(len(type_.shape)))
            for index, held in enumerate(axes)
            if held == axis
        )
        result = types.Layout(
            tuple(local.shape[index] for index in order),
            tuple(local.strides[index] for index in order),
        )
        if shard is None:
            return result
        positions = {old: new for new, old in enumerate(order)}
        attrs = tuple(
            types.Split(positions[attr.axis]) if isinstance(attr, types.Split) else attr for attr in shard.attrs
        )
        return replace(shard, layout=result, attrs=attrs)

    return replace(type_, shape=tuple(reversed(type_.shape)), layout=permuted(type_.layout))


@interfaces_schedule.register_candidates(nn.MatMul, (tir_nn.TiledMma,))
def matmul_operands(call, instruction):
    """Map HIR physical matrix axes to the instruction's fixed M,K and K,N axes."""
    op = call.target
    lhs = matrix_type(call.args[0].type, op.a_layout == "KM")
    rhs = matrix_type(call.args[1].type, op.b_layout == "NK")
    return {}, (call.type, lhs, rhs)


interfaces_schedule.register_candidates(
    sharding.Reshard, (tir.async_copy.CopyAsync, memory.CopyAsyncTensor, memory.CopyReduceAsyncTensor, memory.LdMatrix, memory.StMatrix), when=schedule.only_moves
)


@interfaces_schedule.register_candidates(tensor.IndexSelect, (tir.async_copy.CopyAsync,))
def index_select_operands(call, instruction):
    """Supply the table, result and index; the HIR fill value belongs to the copy."""
    return {"fill": call.target.fill_value}, (call.args[0].type, call.type, call.args[1].type)


__all__: list[str] = []
