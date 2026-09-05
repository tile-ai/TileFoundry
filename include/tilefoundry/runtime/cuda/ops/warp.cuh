/// tilefoundry warp ops — the warp-scoped exchange primitives.
///
/// Included IN-CONTEXT from runtime.cuh, inside ``namespace
/// tilefoundry::ops``: it opens no namespace and pulls in no system headers.
/// Included **before** ``ops/reduce.cuh`` so reduce's intra-warp tier folds
/// through ``warp_reduce`` rather than spelling the butterfly a second time.

/// Three ops, one public entry each, and no tiers: the operand is a
/// register-resident scalar and carries no ``ShardLayout``, so a compile-time
/// tier selection would have nothing to read. [runtime
/// §3](docs/spec/runtime.md#3-runtime-ops) asks for one entry per op, not for
/// every op to have more than one implementation.
#pragma once

#include "warp/warp_impl.h"

/// Exchange ``value`` with the lane whose id differs from this one in the bits
/// of ``lane_mask``; ``member_mask`` names the lanes that participate.
///
/// A butterfly step: after ``k`` steps with masks 1, 2, ... 2^(k-1) every lane
/// holds the combination of its 2^k-lane block.
template <class T>
__device__ inline T shuffle_xor(T value, int lane_mask,
                                unsigned member_mask = 0xFFFFFFFFu) {
    return warp_impl::ShuffleXor<T>{}(value, lane_mask, member_mask);
}

/// True on exactly **one** thread of the leading ``Width`` threads of the CTA.
///
/// ``Width`` is the participating thread count, not a lane mask: a whole block
/// still yields one thread, in the first warp. That is what an mbarrier whose
/// arrive-count is 1 needs — one arriver, chosen without a ballot round-trip.
template <int Width> __device__ inline bool shuffle_elect() {
    return warp_impl::Elect<Width>{}();
}

/// Fold ``value`` across the 32 lanes with ``Combine``, leaving the result in
/// **every** lane.
///
/// ``Combine`` is a functor type with a static ``apply(T, T) -> T``. The fold
/// is the five-step butterfly, so the instruction count to expect is five
/// shuffles and five combines, not a loop over 32 lanes.
template <class Combine, class T> __device__ inline T warp_reduce(T value) {
    return warp_impl::Butterfly<Combine, T>{}(value);
}

/// Sum combine for ``warp_reduce``.
struct warp_sum {
    __device__ static float apply(float a, float b) { return a + b; }
};

/// Max combine for ``warp_reduce``.
struct warp_max {
    __device__ static float apply(float a, float b) { return fmaxf(a, b); }
};

/// Min combine for ``warp_reduce``.
struct warp_min {
    __device__ static float apply(float a, float b) { return fminf(a, b); }
};
