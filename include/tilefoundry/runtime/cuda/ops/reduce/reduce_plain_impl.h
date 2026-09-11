/// reduce plain (non-sharded) rank-aware fold.
///
/// Included in-context from ``ops/reduce.cuh`` (see reduce_common_impl.h for
/// the in-context include contract). Op tags and reduce_impl helpers are in
/// scope.
#pragma once

namespace reduce_impl {

/// Fold what this instance holds, and stop.
template <class Op, class Axes> struct Plain {
    template <class SrcT, class DstT>
    __device__ void operator()(SrcT const &src, DstT &dst) const {
        static_assert(is_supported_reduce_op_v<Op>,
                      "tilefoundry::ops::reduce: unsupported Op");
        auto s = tilefoundry::local_tensor(src);
        auto &&d = tilefoundry::local_tensor(dst);
        using value_type = cute::remove_cvref_t<decltype(d(0))>;
        constexpr int kCells = kept_cells<Axes, decltype(s)>();
        constexpr int kSpan = reduced_span<Axes, decltype(s)>();
        CUTE_UNROLL
        for (int j = 0; j < kCells; ++j) {
            const float acc = local_fold<Op, Axes>(s, j);
            d(j) = static_cast<value_type>(
                reduce_traits<Op>::finalize(acc, float(kSpan)));
        }
    }
};

}
