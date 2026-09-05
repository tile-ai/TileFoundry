/// tilefoundry mbarrier ops — the Hopper shared-memory barrier object.
///
/// Included IN-CONTEXT from runtime.cuh inside ``namespace tilefoundry::ops``;
/// it opens no namespace and pulls in no system headers. The object itself is a
/// 64-bit word in shared memory, and every entry takes a generic pointer to it
/// and converts to the shared window address the instructions want.

/// **Why these are separate entries and ``sync`` is one.** ``ops::sync``
/// collapses five ``SyncKind``s behind one entry because they are five ways to
/// spell one op — "barrier for this mesh scope" — with the scope choosing the
/// spelling. ``mbarrier.init`` / ``arrive`` / ``try_wait`` are three
/// instructions with three meanings and three signatures.

/// [runtime §3](docs/spec/runtime.md#3-runtime-ops)'s one-entry rule forbids
/// exposing the *tiers* of one op separately; it does not ask distinct
/// operations to hide behind an enum. **No tiers here:** a barrier is a
/// shared-memory object, not a sharded tensor, so no operand carries a
/// ``ShardLayout`` for a trait to read.
#pragma once

#include "mbarrier/mbarrier_impl.h"

/// Initialise ``bar`` so that ``arrive_count`` arrivals complete a phase.
///
/// One thread must call this, and a ``sync`` covering every thread that will
/// use the barrier must follow it before any of them arrive or wait.
__device__ inline void mbarrier_init(uint64_t *bar, unsigned arrive_count) {
    mbarrier_impl::Init{}(bar, arrive_count);
}

/// Arrive on ``bar``, contributing one of the arrivals its phase is waiting on.
__device__ inline void mbarrier_arrive(uint64_t *bar) {
    mbarrier_impl::Arrive{}(bar);
}

/// Arrive on ``bar`` **and** declare that ``tx_bytes`` of asynchronous data
/// will land against this phase.
///
/// This is the producer half of a bulk copy: the phase completes when both the
/// arrivals and the byte count are satisfied, so the consumer's wait covers the
/// copy without a second barrier. One instruction rather than an ``arrive``
/// plus an ``expect_tx``.
__device__ inline void mbarrier_arrive_expect_tx(uint64_t *bar,
                                                 unsigned tx_bytes) {
    mbarrier_impl::ArriveExpectTx{}(bar, tx_bytes);
}

/// Declare ``tx_bytes`` of asynchronous data against ``bar``'s current phase
/// without arriving on it.
__device__ inline void mbarrier_expect_tx(uint64_t *bar, unsigned tx_bytes) {
    mbarrier_impl::ExpectTx{}(bar, tx_bytes);
}

/// Test ``bar``'s phase once, without blocking; true when ``phase`` has
/// flipped.
__device__ inline bool mbarrier_try_wait_parity(uint64_t *bar, unsigned phase) {
    return mbarrier_impl::TryWaitParity{}(bar, phase);
}

/// Block until ``bar``'s phase parity reaches ``phase``.
///
/// ``phase`` alternates 0, 1, 0, ... across successive completions of the same
/// barrier, which is what lets a pipeline reuse a fixed ring of barriers
/// instead of allocating one per stage.
__device__ inline void mbarrier_wait_parity(uint64_t *bar, unsigned phase) {
    mbarrier_impl::WaitParity{}(bar, phase);
}

/// Invalidate ``bar``, releasing the shared-memory word for other use.
__device__ inline void mbarrier_invalidate(uint64_t *bar) {
    mbarrier_impl::Invalidate{}(bar);
}
