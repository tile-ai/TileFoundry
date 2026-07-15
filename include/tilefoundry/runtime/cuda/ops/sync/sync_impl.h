// CUDA sync op implementation. Included in-context from ops/sync.cuh inside
// namespace tilefoundry::ops.
#pragma once

namespace sync_impl {

__device__ __forceinline__ void grid_barrier(unsigned int *bar) {
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned int n_ctas = gridDim.x * gridDim.y * gridDim.z;
        unsigned int phase = atomicAdd(&bar[1], 0u);
        __threadfence();
        unsigned int arrived = atomicAdd(&bar[0], 1u) + 1u;
        if (arrived == n_ctas) {
            bar[0] = 0u;
            __threadfence();
            atomicAdd(&bar[1], 1u);
        } else {
            while (atomicAdd(&bar[1], 0u) == phase) {
            }
        }
    }
    __syncthreads();
}

template <auto Kind, int Base, int Count, unsigned Mask, int BarId>
struct Sync {
    __device__ void operator()(unsigned int *grid_bar) const {
        if constexpr (Kind == SyncKind::grid) {
            grid_barrier(grid_bar);
        } else if constexpr (Kind == SyncKind::syncthreads) {
            __syncthreads();
        } else if constexpr (Kind == SyncKind::syncwarp_full) {
            __syncwarp();
        } else if constexpr (Kind == SyncKind::syncwarp_masked) {
            const int tid = int(
                tilefoundry::program_id<tilefoundry::TopologyScope::thread>());
            if (tid >= Base && tid < Base + Count)
                __syncwarp(Mask);
        } else if constexpr (Kind == SyncKind::bar_sync) {
            const int tid = int(
                tilefoundry::program_id<tilefoundry::TopologyScope::thread>());
            if (tid >= Base && tid < Base + Count)
                asm volatile("bar.sync %0, %1;" ::"r"(BarId), "r"(Count));
        }
    }
};

} // namespace sync_impl
