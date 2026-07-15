// CUDA binary op implementation. Included in-context from ops/binary.cuh
// inside namespace tilefoundry::ops.
#pragma once

namespace binary_impl {

template <class Op> struct Binary {
    template <class TL, class TR, class TOut>
    CUTE_HOST_DEVICE void operator()(TL const &lhs, TR const &rhs, TOut &dst,
                                     int N, Op op = {}) const {
        auto l = detail::to_local(lhs);
        auto r = detail::to_local(rhs);
        auto &&d = detail::to_local(dst);
        using value_type = cute::remove_cvref_t<decltype(d(0))>;
        for (int i = 0; i < N; ++i) {
            d(i) = static_cast<value_type>(op(static_cast<value_type>(l(i)),
                                              static_cast<value_type>(r(i))));
        }
    }
};

template <class Op> struct BinaryCellBcast {
    template <class TL, class TR, class TOut>
    CUTE_HOST_DEVICE void operator()(TL const &lhs, TR const &rhs, TOut &dst,
                                     int n_dst, int step, Op op = {}) const {
        auto l = detail::to_local(lhs);
        auto r = detail::to_local(rhs);
        auto &&d = detail::to_local(dst);
        using value_type = cute::remove_cvref_t<decltype(d(0))>;
        constexpr int l_rank = decltype(cute::rank(l))::value;
        constexpr int d_rank = decltype(cute::rank(d))::value;
        for (int j = 0; j < n_dst; ++j) {
            auto rj = static_cast<value_type>(r(j));
            if constexpr (l_rank > 1 && d_rank > 1) {
                for (int k = 0; k < step; ++k) {
                    d(j, k) = static_cast<value_type>(
                        op(static_cast<value_type>(l(j, k)), rj));
                }
            } else {
                for (int k = 0; k < step; ++k) {
                    const int i = j * step + k;
                    d(i) = static_cast<value_type>(
                        op(static_cast<value_type>(l(i)), rj));
                }
            }
        }
    }
};

template <class Op> struct BinaryBcastCol {
    template <class TL, class TR, class TOut>
    CUTE_HOST_DEVICE void operator()(TL const &lhs, TR const &rhs, TOut &dst,
                                     int M, int K, Op op = {}) const {
        using value_type = cute::remove_cvref_t<decltype(dst(0))>;
        for (int m = 0; m < M; ++m) {
            auto scale = static_cast<value_type>(rhs(m, 0));
            for (int k = 0; k < K; ++k) {
                dst(m, k) = static_cast<value_type>(
                    op(static_cast<value_type>(lhs(m, k)), scale));
            }
        }
    }
};

template <class Op> struct BinaryBcastRow {
    template <class TL, class TR, class TOut>
    CUTE_HOST_DEVICE void operator()(TL const &lhs, TR const &rhs, TOut &dst,
                                     int M, int K, Op op = {}) const {
        using value_type = cute::remove_cvref_t<decltype(dst(0))>;
        for (int m = 0; m < M; ++m) {
            for (int k = 0; k < K; ++k) {
                dst(m, k) = static_cast<value_type>(
                    op(static_cast<value_type>(lhs(m, k)),
                       static_cast<value_type>(rhs(k))));
            }
        }
    }
};

template <class Op> struct BinaryBcastScalar {
    template <class TS, class TV, class TOut>
    CUTE_HOST_DEVICE void operator()(TS const &src, TV const &scalar, TOut &dst,
                                     int N, Op op = {}) const {
        auto s = detail::to_local(src);
        auto sc = detail::to_local(scalar);
        auto &&d = detail::to_local(dst);
        using value_type = cute::remove_cvref_t<decltype(d(0))>;
        auto val = static_cast<value_type>(sc(0));
        for (int i = 0; i < N; ++i) {
            d(i) =
                static_cast<value_type>(op(static_cast<value_type>(s(i)), val));
        }
    }
};

} // namespace binary_impl
