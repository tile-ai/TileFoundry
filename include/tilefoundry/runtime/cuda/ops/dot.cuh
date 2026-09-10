/// CUDA dot public entry included in tilefoundry::ops.
#pragma once

#include "dot/dot_impl.h"

/// ``dst = sum(lhs * rhs)`` over the axes the operands' meshes contract.
template <class Lhs, class Rhs, class Dst,
          class Ws = reduce_impl::no_workspace_t>
__device__ inline void dot(Lhs const &lhs, Rhs const &rhs, Dst &dst,
                           Ws &&ws = {}) {
    if constexpr (std::is_same_v<cute::remove_cvref_t<Ws>,
                                 reduce_impl::no_workspace_t>) {
        dot_impl::Warp{}(lhs, rhs, dst);
    } else {
        dot_impl::Cta{}(lhs, rhs, dst, ws);
    }
}
