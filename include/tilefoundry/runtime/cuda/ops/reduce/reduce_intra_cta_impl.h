/// reduce tier-2: cross-warp within a CTA (smem workspace).
///
/// Included in-context from ``ops/reduce.cuh`` (see reduce_common_impl.h for
/// the in-context include contract). Op tags and reduce_impl helpers are in
/// scope.
#pragma once

namespace reduce_impl {

/// ── Reduce tier-2: cross-warp within a CTA (smem workspace) ───────
/// This tier's contract has a single group output per thread (the reduced
/// ``n_cells == 1``): each thread folds its *entire* per-thread tensor via
/// ``local_fold`` (its ``step == cute::size(s)`` whole-tensor path — correct
/// however many cute axes the per-thread layout carries, see
/// reduce_common_impl.h), then a warp butterfly + ``cta_combine_via_workspace``
/// combine across warps; ``reduce_traits<Op>`` finalises. MEAN divides by
/// ``step × 32 × warps_per_group``.
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
        auto s = detail::local_tensor(src);
        auto &&d = detail::local_tensor(dst);
        auto &&ws = detail::local_tensor(workspace);
        using value_type = cute::remove_cvref_t<decltype(d(0))>;

        static_assert(kept_cells<Axes, decltype(s)>() == 1,
                      "ops::reduce (intra-CTA tier): kept axes must form "
                      "one cell");
        constexpr int kSpan = reduced_span<Axes, decltype(s)>();
        const float local = local_fold<Op, Axes>(s, 0);
        const float warp_partial =
            tilefoundry::warp_reduce<typename reduce_traits<Op>::combine_op,
                                     plan.lanes_reduced>(local);
        const float cta_partial =
            cta_combine_via_workspace<Op>(warp_partial, ws, warps_per_group);
        const float total_n =
            float(kSpan) * float(plan.lanes_reduced) * float(warps_per_group);
        const float result = reduce_traits<Op>::finalize(cta_partial, total_n);
        for (int i = 0; i < static_cast<int>(cute::size(d)); ++i) {
            d(i) = static_cast<value_type>(result);
        }
    }
};

}
