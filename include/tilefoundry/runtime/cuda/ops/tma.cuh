/// tilefoundry TMA op — one public entry, the tier read off the shard layouts.
///
/// Included IN-CONTEXT from runtime.cuh inside ``namespace tilefoundry::ops``;
/// it opens no namespace and pulls in no system headers.

/// The entry takes tensors, not addresses. A caller that had to pass a byte
/// count and a raw pointer would be deciding the vector width, the row stride
/// and the bounds itself -- which is the layout's job, and the reason
/// [runtime §3](docs/spec/runtime.md#3-runtime-ops) puts the tier behind one
/// entry rather than in the call site.
#pragma once

#include "tma/tma_impl.h"

/// Stage ``src`` into ``dst``, completing on ``bar``.
///
/// The tier comes from the operands' ``ShardLayout`` at compile time: a slice
/// that is one contiguous run of bytes takes the bulk instruction, anything
/// else takes the strided path. Both leave ``bar`` completing when the data is
/// readable, so the caller waits on the phase and never learns which ran.
///
/// One thread issues the bulk tier; the strided tier is taken by every thread
/// in the block. Both are safe to call from all of them.
template <class Src, class Dst>
__device__ inline void tma_copy(Src const &src, Dst &dst, uint64_t *bar) {
    if constexpr (tma_impl::bulk_eligible_v<Src, Dst>) {
        tma_impl::Bulk{}(src, dst, bar);
    } else {
        tma_impl::Strided{}(src, dst, bar);
    }
}

/// Whether ``tma_copy`` on these operands will take the bulk instruction.
///
/// Exposed because a caller that is sizing a shared ring wants to know whether
/// the ring is being filled asynchronously, and because a test that means to
/// exercise one tier should be able to say which it got.
template <class Src, class Dst>
inline constexpr bool tma_copy_is_bulk = tma_impl::bulk_eligible_v<Src, Dst>;
