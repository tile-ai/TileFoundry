/// CUDA bulk asynchronous copy public entry.
#pragma once

#include "copy_async_bulk/copy_async_bulk_impl.h"

/// Stage ``src`` into ``dst``, completing on ``bar``.
template <class Src, class Dst>
__device__ inline void copy_async_bulk(Src const &src, Dst &dst,
                                       uint64_t *bar) {
    copy_async_bulk_impl::check_operands<Src, Dst>();
    copy_async_bulk_impl::Bulk{}(src, dst, bar);
}
