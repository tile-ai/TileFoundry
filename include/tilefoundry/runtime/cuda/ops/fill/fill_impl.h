// CUDA fill op implementation. Included in-context from ops/fill.cuh inside
// namespace tilefoundry::ops.
#pragma once

namespace fill_impl {

struct Fill {
    template <class TOut>
    __device__ void operator()(TOut &dst, float val, int N) const {
        auto &&d = detail::to_local(dst);
        using value_type = cute::remove_cvref_t<decltype(d(0))>;
        for (int i = 0; i < N; ++i) {
            d(i) = static_cast<value_type>(val);
        }
    }
};

} // namespace fill_impl
