/// CUDA MMA op public entry. Included in-context from runtime.cuh inside
/// namespace tilefoundry::ops.
#pragma once

#include "mma/mma_impl.h"

/// ``c += a @ b``, one entry, the tier read off the operand layouts.
///
/// Rank-2 static shard layouts on ``a`` and ``b`` are a tile, and the entry
/// loops the atom over it; anything else is a lane's already-gathered fragment
/// and takes the single instruction. Codegen emits this one call either way and
/// never says which ([runtime §3](docs/spec/runtime.md#3-runtime-ops)).
///
/// The tile tier reads ``a`` as ``(M, K)`` and ``b`` as ``(N, K)``. There is no
/// transpose flag: whether the buffer behind ``b`` is k-major or n-major is a
/// stride in its layout, and the indexing picks that up.
template <class TA, class TB, class TC>
__device__ void mma(TA const &a, TB const &b, TC &c) {
    if constexpr (mma_impl::tile_shaped_v<TA, TB, TC>) {
        mma_impl::Tile{}(a, b, c);
    } else {
        mma_impl::Atom{}(a, b, c);
    }
}

/// Whether ``mma`` on these operands loops the atom over a tile.
template <class TA, class TB, class TC>
inline constexpr bool mma_is_tile = mma_impl::tile_shaped_v<TA, TB, TC>;

/// Accumulator values one lane holds for an ``M x N`` tile over ``Threads``.
///
/// Warps split N, so a lane's share is ``M/16 * (N/8/warps)`` atoms of four.
template <int M, int N, int Threads> constexpr int mma_acc_elems() {
    return (M / mma_detail::Sm80_16x8x16::kM) *
           (N / (mma_detail::Sm80_16x8x16::kN * (Threads / 32))) *
           mma_detail::Sm80_16x8x16::kAccPerLane;
}

/// This lane's accumulator for an ``M x N`` tile, as a ShardTensor.
///
/// The engine is the lane's own registers -- ``local()`` on register storage
/// hands back exactly that -- while the shard layout carries the tile shape and
/// the thread mesh, which is where ``mma`` reads the warp count from. A caller
/// that built this by hand would be restating the instruction's fragment
/// length at the call site.
template <int M, int N, int Threads, class Ptr>
__device__ auto mma_acc_tensor(Ptr regs) {
    constexpr int elems = mma_acc_elems<M, N, Threads>();
    auto tile =
        cute::make_layout(cute::make_shape(cute::Int<M>{}, cute::Int<N>{}),
                          cute::make_stride(cute::Int<N>{}, cute::Int<1>{}));
    auto mesh_layout = cute::make_layout(cute::make_shape(cute::Int<Threads>{}),
                                         cute::make_stride(cute::Int<1>{}));
    tilefoundry::Mesh<
        tilefoundry::Topology<tilefoundry::TopologyScope::thread, Threads>,
        decltype(mesh_layout)>
        mesh{mesh_layout};
    tilefoundry::ShardLayout<decltype(tile), cute::tuple<tilefoundry::shard::B>,
                             decltype(mesh)>
        shard{tile, mesh};
    return tilefoundry::make_shard_tensor(
        cute::make_tensor(regs, cute::Int<elems>{}), tile, shard);
}

/// The ``(row, col)`` of accumulator entry ``f`` on thread ``tid``.
///
/// The map belongs to the instruction, so the op publishes it rather than
/// leaving each call site to rediscover it: a caller scaling the accumulator by
/// something per row asks here for the row.
template <int M, int N, int Threads>
__device__ cute::tuple<int, int> mma_acc_coord(int f, int tid) {
    using Geo = mma_detail::Sm80_16x8x16;
    constexpr int warps = Threads / 32;
    constexpr int n_atoms = N / (Geo::kN * warps);
    const int lane = tid & 31;
    const int warp = (tid >> 5) % warps;
    const int atom = f / Geo::kAccPerLane;
    const int v = f % Geo::kAccPerLane;
    const int row = (atom / n_atoms) * Geo::kM + Geo::row(lane) + (v >> 1) * 8;
    const int col =
        (warp * n_atoms + atom % n_atoms) * Geo::kN + Geo::col(lane) + (v & 1);
    return cute::make_tuple(row, col);
}
