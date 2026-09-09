/// CUDA dot op public entry. Included in-context from runtime.cuh inside
/// namespace tilefoundry::ops.
#pragma once

#include "dot/dot_impl.h"

/// ``dst = sum(lhs * rhs)`` over the axes the operands' meshes contract.
///
/// A matrix-vector product's inner loop is one statement, not a multiply
/// followed by a reduction: materialising the product would cost a register per
/// element of the row.
///
/// With no workspace the contraction lives inside a warp; with one it spans the
/// block. Either way every participant leaves holding the total, and the load
/// width is the shard layouts' ([runtime
/// §3](docs/spec/runtime.md#3-runtime-ops)).
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
