// reduce tier-1: intra-warp only, no smem workspace.
//
// Included in-context from ``ops/reduce.cuh`` (see reduce_common_impl.h for the
// in-context include contract). Op tags and reduce_impl helpers are in scope.
#pragma once

namespace reduce_impl {

// ── Reduce tier-1: intra-warp only, no smem workspace ─────────────
// Each thread folds its per-cell slice locally, then a 32-lane
// ``__shfl_xor_sync`` butterfly broadcasts the partial.
//
// Per-thread cell decomposition: the per-thread cute tensor has ``size(s)``
// source elements feeding ``size(d)`` destination cells; the source is treated
// as ``size(d)`` contiguous chunks of ``size(s) / size(d)`` lanes, each chunk
// reducing to one output cell.
template <class Op, class Axes> struct IntraWarp {
    template <class SrcT, class DstT>
    __device__ void operator()(SrcT const &src, DstT &dst) const {
        auto s = detail::to_local(src);
        auto &&d = detail::to_local(dst);
        using value_type = cute::remove_cvref_t<decltype(d(0))>;

        const int n_src = static_cast<int>(cute::size(s));
        const int n_dst = static_cast<int>(cute::size(d));
        const int n_cells = (n_dst == 0) ? 1 : n_dst;
        const int step = n_src / n_cells;

        // Cell-aligned 2-D access (``s(j, k)``) keeps each cell on a single
        // row of the per-thread cute layout, regardless of cute's
        // col-major linearisation. Linear ``s(j*step + k)`` mis-aligns
        // when the coalesced per-thread layout has multiple cute axes
        // (factorised single-axis sugar — spec shard §7.1.2). The
        // single-axis path (1-D ``s``) uses linear access — that's the
        // headline rmsnorm seq_1 case where the per-thread cute layout
        // collapses to a single contiguous dim after coalesce.
        constexpr int s_rank = decltype(cute::rank(s))::value;
        for (int j = 0; j < n_cells; ++j) {
            float local;
            if constexpr (std::is_same_v<Op, absmax_op>) {
                local = 0.f;
                if constexpr (s_rank > 1) {
                    for (int k = 0; k < step; ++k) {
                        local =
                            fmaxf(local, fabsf(static_cast<float>(s(j, k))));
                    }
                } else {
                    const int base = j * step;
                    for (int k = 0; k < step; ++k) {
                        local = fmaxf(local,
                                      fabsf(static_cast<float>(s(base + k))));
                    }
                }
            } else {
                local = 0.f;
                if constexpr (s_rank > 1) {
                    for (int k = 0; k < step; ++k) {
                        local += static_cast<float>(s(j, k));
                    }
                } else {
                    const int base = j * step;
                    for (int k = 0; k < step; ++k) {
                        local += static_cast<float>(s(base + k));
                    }
                }
            }
            float partial;
            if constexpr (std::is_same_v<Op, absmax_op>) {
                partial = reduce_impl::warp_max_butterfly(local);
            } else {
                partial = reduce_impl::warp_sum_butterfly(local);
            }
            if constexpr (std::is_same_v<Op, mean_op>) {
                const float total_n = float(step) * 32.f;
                d(j) = static_cast<value_type>(partial / total_n);
            } else if constexpr (std::is_same_v<Op, sum_op> ||
                                 std::is_same_v<Op, absmax_op>) {
                d(j) = static_cast<value_type>(partial);
            } else {
                static_assert(sizeof(Op) == 0,
                              "tilefoundry::ops::reduce: unsupported Op");
            }
        }
    }
};

} // namespace reduce_impl
