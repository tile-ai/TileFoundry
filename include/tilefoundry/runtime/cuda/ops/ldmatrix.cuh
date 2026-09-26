/// CUDA ldmatrix public entry.
#pragma once

#include "ldmatrix/ldmatrix_impl.h"

/// Load one warp's shared-memory tile into its SM80 MMA A fragment.
template <class Src, class Dst>
__device__ inline void ldmatrix(Src const &src, Dst &dst) {
    ldmatrix_impl::LdMatrix{}(src, dst);
}
