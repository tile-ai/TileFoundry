/// CUDA MMA op public entry. Included in-context from runtime.cuh inside
/// namespace tilefoundry::ops.
#pragma once

#include "mma/mma_impl.h"

/// ``c += a @ b`` on one atom's per-lane fragments.
template <class TA, class TB, class TC>
__device__ void mma(TA const &a, TB const &b, TC &c) {
    static_assert(
        mma_impl::atom_shaped_v<TA, TB, TC>,
        "ops::mma: operands must be one atom's lane fragments, bf16 (8, 4, 4) "
        "or e4m3 (16, 8, 4) with f32 C");
    mma_impl::Atom{}(a, b, c);
}
