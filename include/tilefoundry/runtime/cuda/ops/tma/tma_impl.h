/// TMA op implementation — the single implementation of ``tma_bulk_copy``.
///
/// Included in-context from ``ops/tma.cuh``.
#pragma once

namespace tma_impl {

/// ``cp.async.bulk`` global→shared with mbarrier completion.
///
/// ``.shared::cluster`` names the destination window even though this runtime
/// does not distribute a tile across a cluster: that is the operand form the
/// instruction takes, and a single-CTA cluster is the degenerate case of it.
template <class T> struct BulkG2S {
    __device__ void operator()(T const *src_gmem, T *dst_smem, unsigned count,
                               uint64_t *bar) const {
        static_assert(sizeof(T) <= 16,
                      "tma_bulk_copy: element wider than the 16-byte grain");
        const unsigned bytes = count * unsigned(sizeof(T));
        asm volatile(
            "cp.async.bulk.shared::cluster.global"
            ".mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" ::"r"(
                mbarrier_impl::smem_addr(dst_smem)),
            "l"(src_gmem), "r"(bytes), "r"(mbarrier_impl::smem_addr(bar))
            : "memory");
    }
};

}
