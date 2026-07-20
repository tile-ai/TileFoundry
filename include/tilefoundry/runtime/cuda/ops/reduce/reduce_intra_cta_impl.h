// reduce tier-2: cross-warp within a CTA (smem workspace).
//
// Included in-context from ``ops/reduce.cuh`` (see reduce_common_impl.h for the
// in-context include contract). Op tags and reduce_impl helpers are in scope.
#pragma once

namespace reduce_impl {

// ── Reduce tier-2: cross-warp within a CTA (smem workspace) ───────
// This tier's contract has a single group output per thread (``cell_decomp``'s
// ``n_cells == 1``): each thread folds its *entire* per-thread tensor via
// ``local_fold`` (its ``step == cute::size(s)`` whole-tensor path — correct
// however many cute axes the per-thread layout carries, see
// reduce_common_impl.h), then a warp butterfly + ``cta_sum_via_workspace``
// combine across warps; ``reduce_traits<Op>`` finalises. MEAN divides by
// ``step × 32 × warps_per_group``.
template <class Op, class Axes> struct IntraCta {
    template <class SrcT, class DstT, class WorkspaceT>
    __device__ void operator()(SrcT const &src, DstT &dst,
                               WorkspaceT &workspace,
                               int warps_per_group) const {
        static_assert(is_supported_reduce_op_v<Op>,
                      "tilefoundry::ops::reduce: unsupported Op");
        auto s = detail::to_local(src);
        auto &&d = detail::to_local(dst);
        auto &&ws = detail::to_local(workspace);
        using value_type = cute::remove_cvref_t<decltype(d(0))>;

        const auto decomp = cell_decomp(s, d);
        const float local = local_fold<Op>(s, 0, decomp.step);
        const float warp_partial = warp_butterfly<Op>(local);
        const float cta_partial =
            cta_sum_via_workspace(warp_partial, ws, warps_per_group);
        const float total_n =
            float(decomp.step) * 32.f * float(warps_per_group);
        const float result = reduce_traits<Op>::finalize(cta_partial, total_n);
        for (int i = 0; i < static_cast<int>(cute::size(d)); ++i) {
            d(i) = static_cast<value_type>(result);
        }
    }
};

} // namespace reduce_impl
