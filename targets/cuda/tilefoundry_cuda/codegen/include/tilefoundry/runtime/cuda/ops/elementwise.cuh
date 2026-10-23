/// CUDA elementwise public entry included in tilefoundry::ops.
#pragma once

#include "elementwise/elementwise_impl.h"

/// Pointwise: ``dst(i) = fn(src(i)...)`` over the destination's local domain.
template <class Fn, class TOut, class... TIn>
__device__ void elementwise(TOut &dst, Fn fn, TIn const &...src) {
    elementwise_impl::Elementwise{}(dst, fn, src...);
}
