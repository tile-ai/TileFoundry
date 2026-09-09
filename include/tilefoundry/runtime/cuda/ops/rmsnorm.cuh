/// CUDA RMSNorm op public entry. Included in-context from runtime.cuh inside
/// namespace tilefoundry::ops.
///
/// An op and not a composition. The row sum of squares feeds the rescale of
/// that same row: one dependency chain, so the fused form carries the whole
/// row's state in one scalar accumulator. Spelled as a reduce plus a pointwise
/// pass it must materialise that state instead -- an ``M * K`` scratch *per
/// instance*, 16KB a thread at hidden 4096, which is local memory and not
/// registers. The arithmetic is the same; what differs is what it costs.
#pragma once

#include "rmsnorm/rmsnorm_impl.h"

template <class TIn, class TOut, class TW>
__device__ void rmsnorm(TIn const &src, TOut &dst, TW const &weight,
                        float eps) {
    rmsnorm_impl::RmsNorm{}(src, dst, weight, eps);
}
