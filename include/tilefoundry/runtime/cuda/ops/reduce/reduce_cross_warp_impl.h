// reduce tier-2b: cross-warp ONLY (lanes independent).
//
// Included in-context from ``ops/reduce.cuh`` (see reduce_common_impl.h for the
// in-context include contract). Op tags and reduce_impl helpers are in scope.
#pragma once

namespace reduce_impl {

// ── Reduce tier-2b: cross-warp ONLY (lanes independent) ───────────
// Stages one slot per (warp, lane, cell): each thread folds its local cells and
// writes them, and after one ``__syncthreads`` folds its group's warps for its
// own lane.
template <class Op, class Axes> struct CrossWarp {
    template <class SrcT, class DstT, class WorkspaceT>
    __device__ void operator()(SrcT const &src, DstT &dst,
                               WorkspaceT &workspace,
                               int warps_per_group) const {
        auto s = detail::to_local(src);
        auto &&d = detail::to_local(dst);
        auto &&ws = detail::to_local(workspace);
        using value_type = cute::remove_cvref_t<decltype(d(0))>;
        const int n_src = static_cast<int>(cute::size(s));
        const int n_dst = static_cast<int>(cute::size(d));
        const int n_cells = (n_dst == 0) ? 1 : n_dst;
        const int local_n = n_src / n_cells;
        const int lane = threadIdx.x & 31;
        const int warp_id = threadIdx.x >> 5;
        for (int j = 0; j < n_cells; ++j) {
            float acc;
            if constexpr (std::is_same_v<Op, rmax_op>) {
                acc = -INFINITY;
                for (int k = 0; k < local_n; ++k)
                    acc = fmaxf(acc, static_cast<float>(s(j * local_n + k)));
            } else {
                acc = 0.f;
                for (int k = 0; k < local_n; ++k)
                    acc += static_cast<float>(s(j * local_n + k));
            }
            ws((warp_id * 32 + lane) * n_cells + j) = acc;
        }
        __syncthreads();
        const int group_start = (warp_id / warps_per_group) * warps_per_group;
        for (int j = 0; j < n_cells; ++j) {
            float acc;
            if constexpr (std::is_same_v<Op, rmax_op>) {
                acc = -INFINITY;
                for (int w = 0; w < warps_per_group; ++w)
                    acc = fmaxf(
                        acc, static_cast<float>(ws(
                                 ((group_start + w) * 32 + lane) * n_cells + j)));
            } else if constexpr (std::is_same_v<Op, sum_op>) {
                acc = 0.f;
                for (int w = 0; w < warps_per_group; ++w)
                    acc += static_cast<float>(
                        ws(((group_start + w) * 32 + lane) * n_cells + j));
            } else {
                static_assert(std::is_same_v<Op, rmax_op> ||
                                  std::is_same_v<Op, sum_op>,
                              "reduce_cross_warp: only SUM / MAX supported");
            }
            d(j) = static_cast<value_type>(acc);
        }
    }
};

} // namespace reduce_impl
