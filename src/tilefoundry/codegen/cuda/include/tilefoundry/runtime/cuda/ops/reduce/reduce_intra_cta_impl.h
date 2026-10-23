/// reduce tier-2: cross-warp within a CTA (smem workspace).
///
/// Included in-context from ``ops/reduce.cuh`` (see reduce_common_impl.h for
/// the in-context include contract). Op tags and reduce_impl helpers are in
/// scope.
#pragma once

namespace reduce_impl {

/// ── Reduce tier-2: cross-warp within a CTA (smem workspace) ───────
/// This tier's contract has one group output per kept cell: each thread folds
/// its per-thread tensor cell by cell via ``local_fold``, then a warp
/// butterfly + ``cta_combine_via_workspace`` combine across warps, reusing the
/// one-slot-per-warp workspace from cell to cell; ``reduce_traits<Op>``
/// finalises. MEAN divides by ``step × lanes_reduced × warps_per_group``.
template <class Op, class Axes> struct IntraCta {
    template <class SrcT, class DstT, class WorkspaceT>
    __device__ void operator()(SrcT const &src, DstT &dst,
                               WorkspaceT &workspace,
                               int warps_per_group) const {
        constexpr reduce_dispatch_info plan = reduce_dispatch<
            typename cute::remove_cvref_t<SrcT>::shard_layout_type,
            typename cute::remove_cvref_t<DstT>::shard_layout_type>();
        static_assert(is_supported_reduce_op_v<Op>,
                      "tilefoundry::ops::reduce: unsupported Op");
        auto s = tilefoundry::local_tensor(src);
        auto &&d = tilefoundry::local_tensor(dst);
        auto &&ws = tilefoundry::local_tensor(workspace);
        using value_type = cute::remove_cvref_t<decltype(d(0))>;

        constexpr int kCells = kept_cells<Axes, decltype(s)>();
        constexpr int kSpan = reduced_span<Axes, decltype(s)>();
        CUTE_UNROLL
        for (int j = 0; j < kCells; ++j) {
            const float local = local_fold<Op, Axes>(s, j);
            const float warp_partial =
                tilefoundry::warp_reduce<typename reduce_traits<Op>::combine_op,
                                         plan.lanes_reduced>(local);
            const float cta_partial = cta_combine_via_workspace<Op>(
                warp_partial, ws, warps_per_group);
            const float total_n = float(kSpan) * float(plan.lanes_reduced) *
                                  float(warps_per_group);
            d(j) = static_cast<value_type>(
                reduce_traits<Op>::finalize(cta_partial, total_n));
        }
    }
};

}
