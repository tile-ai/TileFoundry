/// CUDA MMA op public entry. Included in-context from runtime.cuh inside
/// namespace tilefoundry::ops.
#pragma once

#include "mma/mma_impl.h"

/// ``c += a @ b``, one entry, the tier read off the operand layouts.
template <class TA, class TB, class TC>
__device__ void mma(TA const &a, TB const &b, TC &c) {
    if constexpr (mma_impl::tile_shaped_v<TA, TB, TC>) {
        mma_impl::Tile{}(a, b, c);
    } else if constexpr (mma_impl::atom_shaped_v<TA, TB, TC>) {
        mma_impl::Atom{}(a, b, c);
    } else {
        static_assert(dependent_false_v<TA>,
                      "ops::mma: the operands are neither a rank-2 static tile "
                      "nor the atom's own (8, 4, 4) lane fragments");
    }
}
