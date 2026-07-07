// CUDA clamp op implementation. Included in-context from ops/clamp.cuh inside
// namespace tilefoundry::ops.
#pragma once

namespace clamp_impl {

struct Clamp {
    template <class TIn, class TOut>
    CUTE_HOST_DEVICE void operator()(TIn const &src, TOut &dst, int N,
                                     float min_val, float max_val) const {
        auto s = detail::to_local(src);
        auto &&d = detail::to_local(dst);
        using value_type = cute::remove_cvref_t<decltype(d(0))>;
        auto mn = static_cast<value_type>(min_val);
        auto mx = static_cast<value_type>(max_val);
        for (int i = 0; i < N; ++i) {
            auto v = static_cast<value_type>(s(i));
            d(i) = (v < mn) ? mn : ((v > mx) ? mx : v);
        }
    }
};

} // namespace clamp_impl
