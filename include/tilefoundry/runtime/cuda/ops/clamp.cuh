// CUDA clamp op public entry and tag. Included in-context from runtime.cuh
// inside namespace tilefoundry::ops.
#pragma once

struct clamp_op {
    float min_val, max_val;
    template <class T> __device__ T operator()(T x) const {
        return x < static_cast<T>(min_val)
                   ? static_cast<T>(min_val)
                   : (x > static_cast<T>(max_val) ? static_cast<T>(max_val)
                                                  : x);
    }
};

#include "clamp/clamp_impl.h"

template <class TIn, class TOut>
CUTE_HOST_DEVICE void clamp(TIn const &src, TOut &dst, int N, float min_val,
                            float max_val) {
    clamp_impl::Clamp{}(src, dst, N, min_val, max_val);
}
