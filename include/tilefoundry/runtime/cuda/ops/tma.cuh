/// tilefoundry TMA op — one public entry, the operands checked against it.

/// TMA derives addresses and byte counts from tensor layouts.
#pragma once

#include "tma/tma_impl.h"

/// Stage ``src`` into ``dst``, completing on ``bar``.
template <class Src, class Dst>
__device__ inline void tma_copy(Src const &src, Dst &dst, uint64_t *bar) {
    tma_impl::check_tma_operands<Src, Dst>();
    tma_impl::Bulk{}(src, dst, bar);
}
