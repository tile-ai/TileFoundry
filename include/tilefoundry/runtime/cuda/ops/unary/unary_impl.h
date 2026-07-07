// CUDA unary op implementation. Included in-context from ops/unary.cuh inside
// namespace tilefoundry::ops.
#pragma once

namespace unary_impl {

template <class Op> struct Unary {
    template <class TIn, class TOut>
    CUTE_HOST_DEVICE void operator()(TIn const &src, TOut &dst, int N,
                                     Op op = {}) const {
        auto s = detail::to_local(src);
        auto &&d = detail::to_local(dst);
        using value_type = cute::remove_cvref_t<decltype(d(0))>;
        for (int i = 0; i < N; ++i) {
            d(i) = static_cast<value_type>(op(s(i)));
        }
    }
};

} // namespace unary_impl
