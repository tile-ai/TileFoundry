/// CUDA RMSNorm op public entry. Included in-context from runtime.cuh inside
/// namespace tilefoundry::ops.
/// CUDA RMSNorm public entry.
#pragma once

#include "rmsnorm/rmsnorm_impl.h"

template <class TIn, class TOut, class TW>
__device__ void rmsnorm(TIn const &src, TOut &dst, TW const &weight,
                        float eps) {
    rmsnorm_impl::RmsNorm{}(src, dst, weight, eps);
}
