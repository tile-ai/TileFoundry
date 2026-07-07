// CUDA RMSNorm op public entry. Included in-context from runtime.cuh inside
// namespace tilefoundry::ops.
#pragma once

#include "rmsnorm/rmsnorm_impl.h"

template <class TIn, class TOut, class TW>
CUTE_HOST_DEVICE void rmsnorm(TIn const &src, TOut &dst, TW const &weight,
                              int M, int K, float eps) {
    rmsnorm_impl::RmsNorm{}(src, dst, weight, M, K, eps);
}
