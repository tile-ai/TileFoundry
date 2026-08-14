from __future__ import annotations

from tilefoundry.module import module
from tilefoundry import func
from tilefoundry.dsl.tf import *  # noqa: F401, F403
from tilefoundry.dsl import Tensor
from tilefoundry.dsl.storage import gmem, host, rmem, smem, tmem  # noqa: F401
from tilefoundry.ir.types.shard import (
    B, S, P, ComposedLayout, Layout, Mesh, ShardLayout, Topology,
)
from tilefoundry.ir.types.dim import DimVar, ceildiv

seq_len = DimVar("seq_len", 1, 100)

@module(entry="single_carry")
class HirGrid:
    @func
    def range_default_step(
        x: Tensor[(8,), "f32"]
    ) -> Tensor[(8,), "f32"]:
        for i in range(8):
            y = relu(x)
        return ()

    @func
    def tile_extent_step(
        x: Tensor[(8,), "f32"]
    ) -> Tensor[(8,), "f32"]:
        for i in range(0, 8, 2):
            y = relu(x)
        return ()

    @func
    def tile_dimvar_extent(
        x: Tensor[(seq_len, 4), "f32"]
    ) -> Tensor[(seq_len, 4), "f32"]:
        for i in range(0, seq_len, 2):
            y = relu(x)
        return ()

    @func
    def range_dim_expr_extent(
        x: Tensor[(seq_len, 4), "f32"]
    ) -> Tensor[(seq_len, 4), "f32"]:
        for i in range(seq_len // 2):
            y = relu(x)
        return ()

    @func
    def range_start_stop_step(
        x: Tensor[(8,), "f32"]
    ) -> Tensor[(8,), "f32"]:
        for i in range(2, 8, 3):
            y = relu(x)
        return ()

    @func
    def inner_bindings_carry_nothing(
        x: Tensor[(8,), "f32"]
    ) -> Tensor[(8,), "f32"]:
        for i in range(8):
            t = relu(x)
            z = add(t, x)
        return ()

    @func
    def carry_reads_old_and_new(
        x: Tensor[(8,), "f32"]
    ) -> Tensor[(8,), "f32"]:
        o = relu(x)
        m = relu(x)
        for i in range(8):
            m_new = max(m, x)
            correction = sub(m, m_new)
            o_2 = add(o, correction)
            o = o_2
            m = m_new
        return o

    @func
    def carry_initialized_from_a_parameter(
        acc: Tensor[(8,), "f32"]
    ) -> Tensor[(8,), "f32"]:
        for i in range(8):
            acc_2 = add(acc, acc)
            acc = acc_2
        return acc

    @func
    def nested_for(
        x: Tensor[(8, 4), "f32"]
    ) -> Tensor[(8, 4), "f32"]:
        o = relu(x)
        for r in range(8):
            for c in range(4):
                o_2 = add(o, x)
                o = o_2
            o = o
        return o

    @func
    def where_on_a_binding(
        x: Tensor[(8, 16), "bf16"]
    ) -> Tensor[(8, 16), "bf16"]:
        y = add(x, x)
        y: where(layout=(_, 16 @ cta), mesh=Mesh((Topology("cta", 8),), Layout((8,), (1,))), storage="gmem")
        return y

    @func
    def where_with_a_partial_value_state(
        x: Tensor[(8, 16), "bf16"]
    ) -> Tensor[(8, 16), "bf16"]:
        y = add(x, x)
        y: where(layout=((_, 16), {cta @ P("sum")}))
        return y

    @func
    def where_on_a_parameter(
        x: Tensor[(8, 16), "bf16"]
    ) -> Tensor[(8, 16), "bf16"]:
        x: where(storage="smem")
        return x

    @func
    def where_on_a_bound_tuple_element(
        x: Tensor[(8, 16), "bf16"]
    ) -> Tensor[(8, 4), "i64"]:
        values = topk(x, k=4, axis=-1, largest=True, sorted=True)
        ids = tuple_get_item(values, index=1)
        ids: where(storage="gmem")
        return ids

    @func
    def single_carry(
        x: Tensor[(8,), "f32"]
    ) -> Tensor[(8,), "f32"]:
        o = relu(x)
        for i in range(8):
            o_2 = add(o, x)
            o = o_2
        return o
