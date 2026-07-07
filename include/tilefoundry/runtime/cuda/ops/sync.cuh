// CUDA sync op public entry. Included in-context from runtime.cuh inside
// namespace tilefoundry::ops.
#pragma once

enum class SyncKind {
    syncthreads,
    syncwarp_full,
    syncwarp_masked,
    bar_sync,
    grid,
};

#include "sync/sync_impl.h"

template <SyncKind Kind, int Base = 0, int Count = 0, unsigned Mask = 0u,
          int BarId = 0>
__device__ inline void sync(unsigned int *grid_bar = nullptr) {
    sync_impl::Sync<Kind, Base, Count, Mask, BarId>{}(grid_bar);
}
