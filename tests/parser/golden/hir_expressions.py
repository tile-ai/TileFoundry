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

CTX_LEN = DimVar("CTX_LEN", 1, 4097)

@module(entry="dim_anchored_twice")
class HirExpressions:
    @func
    def dim_from_a_static_call(
        x: Tensor[(CTX_LEN,), "bf16"]
    ) -> Tensor[((128 * ((CTX_LEN - 1) // 128)) + 128,), "bf16"]:
        v0 = zeros(shape=((128 * ((CTX_LEN - 1) // 128)) + 128,), dtype="bf16", storage=gmem)
        return v0

    @func
    def cast_by_dtype_string(
        x: Tensor[(8,), "f32"]
    ) -> Tensor[(8,), "bf16"]:
        v0 = cast(x, dtype="bf16")
        return v0

    @func
    def reduce_by_kind_string(
        x: Tensor[(8,), "f32"]
    ) -> Tensor[(1,), "f32"]:
        v0 = reduce(x, axes=(0,), keepdim=True, kind="sum")
        return v0

    @func
    def unmaterialized_surface_storage(
        x: Tensor[(8,), "f32", "umat"]
    ) -> Tensor[(8,), "f32"]:
        return x

    @func
    def storage_without_a_layout_slot(
        x: Tensor[(8,), "f32", "umat"]
    ) -> Tensor[(8,), "f32"]:
        return x

    @func
    def literal_meets_bf16(
        x: Tensor[(1, 8), "bf16"]
    ) -> Tensor[(1, 8), "bf16"]:
        v0 = 1e-06
        v1 = add(x, v0)
        return v1

    @func
    def captured_float_meets_bf16(
        x: Tensor[(1, 8), "bf16"]
    ) -> Tensor[(1, 8), "bf16"]:
        v0 = 1e-06
        v1 = add(x, v0)
        return v1

    @func
    def compile_time_operands(
        x: Tensor[(1, 2048), "bf16"]
    ) -> Tensor[(1, 16, 128), "bf16"]:
        v0 = 0.08838834764831845
        scaled = mul(x, v0)
        v1 = 1e-06
        shifted = add(scaled, v1)
        v2 = reshape(shifted, new_shape=(1, 16, 128))
        return v2

    @func
    def unpacked_compile_time_values(
        x: Tensor[(1, 32, 128), "f32"]
    ) -> Tensor[(1, 64, 64), "f32"]:
        v0 = reshape(x, new_shape=(1, 64, 64))
        return v0

    @func
    def offsets_as_a_tuple_literal(
        dst: Tensor[(2, 8, 4), "f32"],
        upd: Tensor[(1, 3, 4), "f32"],
        p: Tensor[(), "i32"]
    ) -> Tensor[(2, 8, 4), "f32"]:
        v0 = 1
        v1 = 0
        v3 = insert_slice(dst, upd, (1, p, 0))
        return v3

    @func
    def unpacked_multi_output(
        x: Tensor[(1, 1536), "bf16"]
    ) -> Tensor[(1, 1536), "fp8e4m3"]:
        quant_out = quant(x, scheme="per_token_group", group=128, target_dtype="fp8e4m3")
        x_fp8 = tuple_get_item(quant_out, index=0)
        return x_fp8

    @func
    def index_drops_its_axis(
        x: Tensor[(1, 4, 8), "f32"]
    ) -> Tensor[(1, 4), "f32"]:
        v0 = 0
        v1 = 0
        v2 = 3
        v4 = x[:, :, 3:4]
        v5 = reshape(v4, new_shape=(1, 4))
        return v5

    @func
    def slice_keeps_its_axis(
        x: Tensor[(1, 4, 8), "f32"]
    ) -> Tensor[(1, 4, 1), "f32"]:
        v0 = 0
        v1 = 0
        v2 = 3
        v4 = x[:, :, 3:4]
        return v4

    @func
    def index_counted_from_the_end(
        x: Tensor[(1, 4, 8), "f32"]
    ) -> Tensor[(1, 4), "f32"]:
        v0 = 0
        v1 = 0
        v2 = 7
        v4 = x[:, :, 7:8]
        v5 = reshape(v4, new_shape=(1, 4))
        return v5

    @func
    def slice_strided_and_clamped(
        x: Tensor[(1, 4, 8), "f32"]
    ) -> Tensor[(1, 4, 3), "f32"]:
        v0 = 0
        v1 = 0
        v2 = 1
        v4 = x[:, :, 1:10:3]
        return v4

    @func
    def slice_to_symbolic_extents(
        x: Tensor[(CTX_LEN, 128), "f32"]
    ) -> Tensor[(CTX_LEN, 128), "f32"]:
        v0 = 0
        v1 = 0
        v3 = x[:, :]
        return v3

    @func
    def full_tile_window(
        x: Tensor[(8, 4), "f32"],
        seed: Tensor[(4, 4), "f32"]
    ) -> Tensor[(4, 4), "f32"]:
        out = add(seed, seed)
        for row in tile(8, 4):
            out_2 = x[row, :]
            out_3 = add(out_2, seed)
            out = out_3
        return out

    @func
    def two_windows_a_fixed_distance_apart(
        gu: Tensor[(3, 8), "f32"],
        seed: Tensor[(3, 2), "f32"]
    ) -> Tensor[(3, 2), "f32"]:
        out = add(seed, seed)
        for n in tile(4, 2):
            out_2 = gu[:, n]
            out_3 = gu[:, n + 4]
            out_4 = mul(out_2, out_3)
            out_5 = add(out, out_4)
            out = out_5
        return out

    @func
    def a_summed_offset_names_the_same_move(
        gu: Tensor[(3, 8), "f32"],
        seed: Tensor[(3, 2), "f32"]
    ) -> Tensor[(3, 2), "f32"]:
        out = add(seed, seed)
        for n in tile(4, 2):
            out_2 = gu[:, n]
            out_3 = gu[:, n + 4]
            out_4 = mul(out_2, out_3)
            out_5 = add(out, out_4)
            out = out_5
        return out

    @func
    def dim_anchored_twice(
        x: Tensor[(CTX_LEN,), "bf16"],
        y: Tensor[(CTX_LEN + 1,), "bf16"]
    ) -> Tensor[(CTX_LEN,), "bf16"]:
        return x
