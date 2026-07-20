// CUDA fill op public entry. Included in-context from runtime.cuh inside
// namespace tilefoundry::ops.
#pragma once

#include "fill/fill_impl.h"

template <class TOut> __device__ void fill(TOut &dst, float val, int N) {
    fill_impl::Fill{}(dst, val, N);
}
