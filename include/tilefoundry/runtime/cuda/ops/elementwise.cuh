/// CUDA elementwise op public entry. Included in-context from runtime.cuh
/// inside namespace tilefoundry::ops.
#pragma once

#include "elementwise/elementwise_impl.h"

/// Pointwise: ``dst(i) = fn(src(i)...)`` over the destination's local domain.
///
/// ``dst`` and ``fn`` lead because the sources are a pack, and the pack's
/// length is the arity: no sources is a fill, one a map, two a combine. ``fn``
/// is anything callable, so the op tags below are already valid values for it
/// and there is no tag path separate from a lambda path. Broadcast is neither
/// an arity nor a name but a stride-0 mode on the operand that broadcasts,
/// which that operand's layout already states ([runtime
/// §3](docs/spec/runtime.md#3-runtime-ops)).
template <class Fn, class TOut, class... TIn>
__device__ void elementwise(TOut &dst, Fn fn, TIn const &...src) {
    elementwise_impl::Elementwise{}(dst, fn, src...);
}
