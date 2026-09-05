/// tilefoundry TMA ops — the Hopper bulk asynchronous copy.
///
/// Included IN-CONTEXT from runtime.cuh inside ``namespace tilefoundry::ops``;
/// it opens no namespace and pulls in no system headers.

/// **One implementation, and why there is no tier.** The hardware's two bulk
/// forms are the rank-1 ``cp.async.bulk`` (a contiguous byte run, no
/// descriptor) and the tensor forms ``cp.async.bulk.tensor.Nd`` (a
/// ``TensorMap`` encoded on the host). Those differ in their *operands* — a
/// size against a descriptor plus coordinates — not in a tier a trait could
/// pick between, and a caller holding one cannot supply the other.

/// This header implements the rank-1 form, the one the address shapes in this
/// repository can use. The tensor form is absent because nothing calls it: an
/// implementation with no caller and no test is a guess, not a contract.

/// A weaker-alignment fallback is likewise absent on purpose — that is
/// ``ops::copy_async``. This op is the bulk instruction, and it says so by
/// refusing rather than by silently becoming a slower copy. Both addresses
/// must be 16-byte aligned and the byte count a multiple of 16; the
/// instruction has no defined behaviour otherwise.
#pragma once

#include "tma/tma_impl.h"

/// Stage ``count`` elements from global memory into shared memory as one bulk
/// asynchronous copy, completing against ``bar``.
///
/// Exactly **one** thread issues this. Completion is signalled on ``bar``'s
/// current phase, so the issuing thread pairs it with
/// ``mbarrier_arrive_expect_tx(bar, bytes)`` and every consumer waits with
/// ``mbarrier_wait_parity``. Nothing here blocks: the point of the op is that
/// the copy is still in flight when the issuing thread reaches the next one.
template <class T>
__device__ inline void tma_bulk_copy(T const *src_gmem, T *dst_smem,
                                     unsigned count, uint64_t *bar) {
    tma_impl::BulkG2S<T>{}(src_gmem, dst_smem, count, bar);
}

/// The byte count ``tma_bulk_copy`` will transfer, for the
/// ``mbarrier_arrive_expect_tx`` that must declare it.
///
/// Spelled as its own function so the two numbers cannot drift: a phase that
/// expects a different byte count than the copy delivers never completes, and
/// that failure looks like a hang rather than like a wrong answer.
template <class T>
__device__ __host__ inline constexpr unsigned tma_bulk_bytes(unsigned count) {
    return count * unsigned(sizeof(T));
}
