// reduce plain (non-sharded) rank-aware fold.
//
// Included in-context from ``ops/reduce.cuh`` (see reduce_common_impl.h for the
// in-context include contract). Op tags are in scope.
#pragma once

namespace reduce_impl {

// Non-sharded reduce over a plain cute tensor. Extents are derived from the
// operand: rank-1 folds all ``cute::size(src)`` elements into ``dst(0)``;
// rank-2 folds each of the ``M`` rows over its ``K`` columns into ``dst(m)``.
// Combine finalisation (``op(acc, count)``) matches the legacy plain reduce.
template <class Op, class Axes> struct Plain {
    template <class SrcT, class DstT>
    __device__ void operator()(SrcT const &src, DstT &dst) const {
        using value_type = cute::remove_cvref_t<decltype(dst(0))>;
        Op op{};
        constexpr int s_rank = decltype(cute::rank(src))::value;
        if constexpr (s_rank == 1) {
            const int N = static_cast<int>(cute::size(src));
            float acc = 0.0f;
            for (int i = 0; i < N; ++i) {
                acc += static_cast<float>(src(i));
            }
            dst(0) = static_cast<value_type>(op(acc, float(N)));
        } else {
            const int M = static_cast<int>(cute::size<0>(src));
            const int K = static_cast<int>(cute::size<1>(src));
            for (int m = 0; m < M; ++m) {
                float acc = 0.0f;
                for (int k = 0; k < K; ++k) {
                    acc += static_cast<float>(src(m * K + k));
                }
                dst(m) = static_cast<value_type>(op(acc, float(K)));
            }
        }
    }
};

} // namespace reduce_impl
