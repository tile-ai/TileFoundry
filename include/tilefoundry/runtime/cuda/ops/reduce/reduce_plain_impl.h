// reduce plain (non-sharded) rank-aware fold.
//
// Included in-context from ``ops/reduce.cuh`` (see reduce_common_impl.h for the
// in-context include contract). Op tags and reduce_impl helpers are in scope.
#pragma once

namespace reduce_impl {

// Non-sharded reduce over a plain cute tensor. Extents are derived from the
// operand via the shared ``cell_decomp``: a scalar ``dst`` folds every
// element of ``src`` into ``dst(0)``; an ``M``-cell ``dst`` folds each of the
// ``M`` cells over its ``size(src) / M`` elements into ``dst(j)``. Combine +
// finalisation come from ``reduce_traits<Op>``, shared with the sharded
// tiers.
template <class Op, class Axes> struct Plain {
    template <class SrcT, class DstT>
    __device__ void operator()(SrcT const &src, DstT &dst) const {
        static_assert(is_supported_reduce_op_v<Op>,
                      "tilefoundry::ops::reduce: unsupported Op");
        using value_type = cute::remove_cvref_t<decltype(dst(0))>;

        const auto decomp = cell_decomp(src, dst);
        for (int j = 0; j < decomp.n_cells; ++j) {
            const float acc = local_fold<Op>(src, j, decomp.step);
            dst(j) = static_cast<value_type>(
                reduce_traits<Op>::finalize(acc, float(decomp.step)));
        }
    }
};

} // namespace reduce_impl
