// reduce tier-2: cross-warp within a CTA (smem workspace).
//
// Included in-context from ``ops/reduce.cuh`` (see reduce_common_impl.h for the
// in-context include contract). Op tags and reduce_impl helpers are in scope.
#pragma once

namespace reduce_impl {

// ── Reduce tier-2: cross-warp within a CTA (smem workspace) ───────
// Each warp folds locally then writes its partial into the workspace; after one
// ``__syncthreads()`` every thread sums its group's slots and broadcasts the
// result to its output cells. ``warps_per_group`` partitions the workspace into
// independent reduction groups. MEAN divides by ``local_n × 32 ×
// warps_per_group``.
template <class Op, class Axes> struct IntraCta {
    template <class SrcT, class DstT, class WorkspaceT>
    __device__ void operator()(SrcT const &src, DstT &dst,
                               WorkspaceT &workspace,
                               int warps_per_group) const {
        auto s = detail::to_local(src);
        auto &&d = detail::to_local(dst);
        auto &&ws = detail::to_local(workspace);
        using value_type = cute::remove_cvref_t<decltype(d(0))>;
        // local_n: elements folded per thread.  Per the tier-1 ``reduce``
        // contract (size(s) % size(d) == 0; size(d) chunks of step lanes),
        // each output cell takes ``size(s) / size(d)`` source elements.
        // For the headline rmsnorm path size(d) == 1 ⇒ local_n == size(s)
        // — folds the entire per-thread tensor. Taking only the last cute
        // axis here (as the prior heuristic did) silently dropped the
        // per-warp residual axes introduced by single-axis sugar
        // factorisation (spec shard §7.1.2 + parser/sugar.py), so the mean
        // reduce summed only a fraction of the row.
        const int n_src = static_cast<int>(cute::size(s));
        const int n_dst = static_cast<int>(cute::size(d));
        const int local_n = (n_dst == 0) ? n_src : (n_src / n_dst);

        float local, warp_partial, cta_partial;
        if constexpr (std::is_same_v<Op, absmax_op>) {
            local = reduce_impl::local_fold_maxabs(s, local_n);
            warp_partial = reduce_impl::warp_max_butterfly(local);
            cta_partial = reduce_impl::cta_sum_via_workspace(warp_partial, ws,
                                                             warps_per_group);
        } else {
            local = reduce_impl::local_fold_sum(s, local_n);
            warp_partial = reduce_impl::warp_sum_butterfly(local);
            cta_partial = reduce_impl::cta_sum_via_workspace(warp_partial, ws,
                                                             warps_per_group);
        }

        if constexpr (std::is_same_v<Op, mean_op>) {
            float total_n = float(local_n) * 32.f * float(warps_per_group);
            float result = cta_partial / total_n;
            for (int i = 0; i < static_cast<int>(cute::size(d)); ++i) {
                d(i) = static_cast<value_type>(result);
            }
        } else if constexpr (std::is_same_v<Op, sum_op>) {
            for (int i = 0; i < static_cast<int>(cute::size(d)); ++i) {
                d(i) = static_cast<value_type>(cta_partial);
            }
        } else if constexpr (std::is_same_v<Op, absmax_op>) {
            for (int i = 0; i < static_cast<int>(cute::size(d)); ++i) {
                d(i) = static_cast<value_type>(cta_partial);
            }
        } else {
            static_assert(sizeof(Op) == 0,
                          "tilefoundry::ops::reduce: unsupported Op");
        }
    }
};

} // namespace reduce_impl
