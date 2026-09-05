/// mbarrier op implementations — one per entry in ``ops/mbarrier.cuh``.
///
/// Included in-context from ``ops/mbarrier.cuh``. Every struct here is the
/// single implementation of its entry; nothing selects a tier.
///
/// The instructions are ``.shared::cta`` throughout: this runtime does not
/// place a barrier in a cluster's distributed shared memory, and naming the
/// window explicitly keeps the generic-to-shared conversion from being redone
/// by the assembler on every use.
#pragma once

namespace mbarrier_impl {

/// The shared-window address of a generic pointer, which is what every
/// ``mbarrier.*`` instruction takes.
__device__ inline uint32_t smem_addr(void const *ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

/// ``mbarrier.init`` — arm the object for ``arrive_count`` arrivals.
struct Init {
    __device__ void operator()(uint64_t *bar, unsigned arrive_count) const {
        asm volatile(
            "mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(smem_addr(bar)),
            "r"(arrive_count));
    }
};

/// ``mbarrier.arrive`` — the arrival token is discarded: this runtime's
/// consumers wait on the phase parity, not on a token handed between threads.
struct Arrive {
    __device__ void operator()(uint64_t *bar) const {
        asm volatile("{\n"
                     "  .reg .b64 state;\n"
                     "  mbarrier.arrive.shared::cta.b64 state, [%0];\n"
                     "}\n" ::"r"(smem_addr(bar)));
    }
};

/// ``mbarrier.arrive.expect_tx`` — arrive and declare the bytes in one go.
struct ArriveExpectTx {
    __device__ void operator()(uint64_t *bar, unsigned tx_bytes) const {
        asm volatile(
            "{\n"
            "  .reg .b64 state;\n"
            "  mbarrier.arrive.expect_tx.shared::cta.b64 state, [%0], %1;\n"
            "}\n" ::"r"(smem_addr(bar)),
            "r"(tx_bytes));
    }
};

/// ``mbarrier.expect_tx`` — declare bytes without contributing an arrival.
struct ExpectTx {
    __device__ void operator()(uint64_t *bar, unsigned tx_bytes) const {
        asm volatile("mbarrier.expect_tx.shared::cta.b64 [%0], %1;\n" ::"r"(
                         smem_addr(bar)),
                     "r"(tx_bytes));
    }
};

/// ``mbarrier.try_wait.parity`` — one non-blocking test of the phase.
///
/// Spelled as a single test with the loop in C++ rather than as a PTX loop with
/// its own labels: a label inside inline asm is emitted once per instantiation
/// and collides as soon as two instantiations land in one translation unit.
struct TryWaitParity {
    __device__ bool operator()(uint64_t *bar, unsigned phase) const {
        unsigned ready = 0u;
        asm volatile(
            "{\n"
            "  .reg .pred complete;\n"
            "  mbarrier.try_wait.parity.shared::cta.b64 complete, [%1], %2;\n"
            "  selp.b32 %0, 1, 0, complete;\n"
            "}\n"
            : "=r"(ready)
            : "r"(smem_addr(bar)), "r"(phase));
        return ready != 0u;
    }
};

/// Spin on ``try_wait`` until the phase flips.
///
/// ``try_wait`` already parks the warp in hardware for a bounded interval, so
/// this loop is not a busy spin on the issue pipe.
struct WaitParity {
    __device__ void operator()(uint64_t *bar, unsigned phase) const {
        while (!TryWaitParity{}(bar, phase)) {
        }
    }
};

/// ``mbarrier.inval`` — release the word.
struct Invalidate {
    __device__ void operator()(uint64_t *bar) const {
        asm volatile(
            "mbarrier.inval.shared::cta.b64 [%0];\n" ::"r"(smem_addr(bar)));
    }
};

}
