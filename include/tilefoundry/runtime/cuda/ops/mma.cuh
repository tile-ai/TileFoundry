// CUDA MMA op public entry. Included in-context from runtime.cuh inside
// namespace tilefoundry::ops.
#pragma once

#include "mma/mma_impl.h"

template <class TA, class TB, class TC>
__device__ void mma_sm80_16x8x16_bf16(TA const &a, TB const &b, TC &c) {
    mma_impl::MmaSm80_16x8x16Bf16{}(a, b, c);
}
