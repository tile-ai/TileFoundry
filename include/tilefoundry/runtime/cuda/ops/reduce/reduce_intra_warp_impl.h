// reduce tier-1: intra-warp only, no smem workspace.
//
// Included in-context from ``ops/reduce.cuh`` (see reduce_common_impl.h for the
// in-context include contract). Op tags and reduce_impl helpers are in scope.
#pragma once

namespace reduce_impl {

// ── Reduce tier-1: intra-warp only, no smem workspace ─────────────
// Each thread folds its per-cell slice locally via ``local_fold`` (rank-aware
// cell decomposition — see reduce_common_impl.h), then a 32-lane
// ``warp_butterfly`` broadcasts the combined partial; ``reduce_traits<Op>``
// finalises (mean divides by the total reduced count).
template <class Op, class Axes> struct IntraWarp {
    template <class SrcT, class DstT>
    __device__ void operator()(SrcT const &src, DstT &dst) const {
        static_assert(is_supported_reduce_op_v<Op>,
                      "tilefoundry::ops::reduce: unsupported Op");
        auto s = detail::to_local(src);
        auto &&d = detail::to_local(dst);
        using value_type = cute::remove_cvref_t<decltype(d(0))>;

        const auto decomp = cell_decomp(s, d);
        for (int j = 0; j < decomp.n_cells; ++j) {
            const float local = local_fold<Op>(s, j, decomp.step);
            const float partial = warp_butterfly<Op>(local);
            const float total_n = float(decomp.step) * 32.f;
            d(j) = static_cast<value_type>(
                reduce_traits<Op>::finalize(partial, total_n));
        }
    }
};

} // namespace reduce_impl
