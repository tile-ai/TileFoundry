/// CUDA bulk asynchronous copy public entry.
#pragma once

#include "copy_async_bulk/copy_async_bulk_impl.h"

/// Stage ``src`` into ``dst`` on the threads of ``ExecutionMesh``,
/// completing on ``bar``.
template <class ExecutionMesh, class Src, class Dst>
__device__ inline void copy_async_bulk(Src const &src, Dst &dst,
                                       uint64_t *bar) {
    copy_async_bulk_impl::check_operands<Src, Dst>();
    copy_async_bulk_impl::Bulk<ExecutionMesh>{}(src, dst, bar);
}
