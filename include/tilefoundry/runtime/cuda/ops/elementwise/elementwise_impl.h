/// CUDA elementwise implementation included in tilefoundry::ops.
#pragma once

namespace elementwise_impl {

/// Apply one callable over the destination's local domain.
struct Elementwise {
    template <class Fn, class TOut, class... TIn>
    __device__ void operator()(TOut &dst, Fn fn, TIn const &...src) const {
        auto &&d = detail::to_local(dst);
        for (int i = 0; i < int(cute::size(d)); ++i) {
            d(i) = fn(detail::to_local(src)(i)...);
        }
    }
};

}
