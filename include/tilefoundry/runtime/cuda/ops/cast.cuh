// CUDA cast op public entry. Included in-context from runtime.cuh inside
// namespace tilefoundry::ops.
#pragma once

#include "cast/cast_impl.h"

template <class TIn, class TOut>
CUTE_HOST_DEVICE void cast(TIn const &src, TOut &dst, int N) {
    cast_impl::Cast{}(src, dst, N);
}
