/// reduce tier-1: intra-warp only, no smem workspace.
///
/// Included in-context from ``ops/reduce.cuh`` (see reduce_common_impl.h for
/// the in-context include contract). Op tags and reduce_impl helpers are in
/// scope.
#pragma once

namespace reduce_impl {

/// ── Reduce tier-1: intra-warp only, no smem workspace ─────────────
/// Each thread folds its per-cell slice locally via ``local_fold`` (rank-aware
/// cell decomposition — see reduce_common_impl.h), then a 32-lane
/// ``warp_reduce`` broadcasts the combined partial; ``reduce_traits<Op>``
/// finalises (mean divides by the total reduced count).
template <class Op, class Axes> struct IntraWarp {
    template <class SrcT, class DstT>
    __device__ void operator()(SrcT const &src, DstT &dst) const {
        static_assert(is_supported_reduce_op_v<Op>,
                      "tilefoundry::ops::reduce: unsupported Op");
        constexpr reduce_dispatch_info plan = reduce_dispatch<
            typename cute::remove_cvref_t<SrcT>::shard_layout_type,
            typename cute::remove_cvref_t<DstT>::shard_layout_type>();
        static_assert(plan.warps_per_group == 1,
                      "ops::reduce (intra-warp tier): this reduce mesh spreads "
                      "one value over more than one warp");
        auto s = detail::to_local(src);
        auto &&d = detail::to_local(dst);
        using value_type = cute::remove_cvref_t<decltype(d(0))>;

        constexpr int kCells = kept_cells<Axes, decltype(s)>();
        constexpr int kSpan = reduced_span<Axes, decltype(s)>();
        CUTE_UNROLL
        for (int j = 0; j < kCells; ++j) {
            const float local = local_fold<Op, Axes>(s, j);
            const float partial =
                tilefoundry::warp_reduce<typename reduce_traits<Op>::combine_op,
                                         plan.lanes_reduced>(local);
            const float total_n = float(kSpan) * float(plan.lanes_reduced);
            d(j) = static_cast<value_type>(
                reduce_traits<Op>::finalize(partial, total_n));
        }
    }
};

}
