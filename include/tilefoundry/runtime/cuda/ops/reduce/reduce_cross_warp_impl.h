/// reduce tier-2b: cross-warp ONLY (lanes independent).
///
/// Included in-context from ``ops/reduce.cuh`` (see reduce_common_impl.h for
/// the in-context include contract). Op tags and reduce_impl helpers are in
/// scope.
#pragma once

namespace reduce_impl {

/// ── Reduce tier-2b: cross-warp ONLY (lanes independent) ───────────
/// Stages one slot per (warp, lane, cell): each thread folds its local cells
/// via the shared rank-aware ``local_fold`` and writes them, and after one
/// ``__syncthreads`` folds its group's warps for its own lane via
/// ``reduce_traits<Op>::combine``; ``finalize`` divides by the total reduced
/// count for MEAN.
template <class Op, class Axes> struct CrossWarp {
    template <class SrcT, class DstT, class WorkspaceT>
    __device__ void operator()(SrcT const &src, DstT &dst,
                               WorkspaceT &workspace,
                               int warps_per_group) const {
        static_assert(is_supported_reduce_op_v<Op>,
                      "tilefoundry::ops::reduce: unsupported Op");
        auto s = detail::local_tensor(src);
        auto &&d = detail::local_tensor(dst);
        auto &&ws = detail::local_tensor(workspace);
        using value_type = cute::remove_cvref_t<decltype(d(0))>;

        constexpr int kCells = kept_cells<Axes, decltype(s)>();
        constexpr int kSpan = reduced_span<Axes, decltype(s)>();
        const int tid = int(program_id<TopologyScope::thread>());
        const int lane = tid % kWarpSize;
        const int warp_id = tid / kWarpSize;
        CUTE_UNROLL
        for (int j = 0; j < kCells; ++j) {
            const float local = local_fold<Op, Axes>(s, j);
            ws((warp_id * kWarpSize + lane) * kCells + j) = local;
        }
        __syncthreads();
        const int group_start = (warp_id / warps_per_group) * warps_per_group;
        CUTE_UNROLL
        for (int j = 0; j < kCells; ++j) {
            float acc = reduce_traits<Op>::init;
            for (int w = 0; w < warps_per_group; ++w) {
                acc = combine<Op>(
                    acc,
                    static_cast<float>(ws(
                        ((group_start + w) * kWarpSize + lane) * kCells + j)));
            }
            const float total_n = float(kSpan) * float(warps_per_group);
            d(j) = static_cast<value_type>(
                reduce_traits<Op>::finalize(acc, total_n));
        }
    }
};

}
