/// CUDA sync op implementation. Included in-context from ops/sync.cuh inside
/// namespace tilefoundry::ops.
#pragma once

namespace sync_impl {

__device__ __forceinline__ void grid_barrier(unsigned int *bar) {
    __syncthreads();
    if (tilefoundry::program_id<tilefoundry::TopologyScope::thread>() == 0) {
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

/// Which barrier a mesh asks for.
enum class Tier {
    grid,
    block,
    warp,
    named,
    illegal_grid_slice,
    illegal_warp_layout,
    illegal_scope
};
struct plan_t {
    Tier tier;
    int base;
    int count;
    bool needs_workspace;
    bool needs_counter;
    bool needs_bar_id;
};
template <Tier, class...> CUTE_HOST_DEVICE constexpr void reject();

/// How many instances a mesh covers, as a compile-time number.
///
/// One level, because every tier below names threads of one block -- a
/// barrier counts those, and a mesh naming a coarser level counts programs
/// no barrier here reaches.
template <class TMesh> CUTE_HOST_DEVICE constexpr int instances() {
    static_assert(TMesh::level_count == 1,
                  "ops::sync: a barrier is over one topology level's "
                  "instances, and this mesh names several");
    return int(cute::size(typename TMesh::layout_type{}));
}

/// Whether the mesh's warps are one unbroken run: ``bar.sync`` names a count
/// of consecutive threads, so a mesh that skips warps has no barrier to name.
///
/// One unbroken run is what ``coalesce`` folds to rank one -- once the axes
/// are in the order CuTe reads them.
template <class TMesh> CUTE_HOST_DEVICE constexpr bool warps_run_together() {
    using ids_t = cute::remove_cvref_t<decltype(detail::id_axes(
        typename TMesh::layout_type{}))>;
    using run_t = cute::remove_cvref_t<decltype(cute::coalesce(ids_t{}))>;
    return decltype(cute::rank(run_t{}))::value == 1 &&
           int(cute::get<0>(cute::stride(run_t{}))) == 1;
}

/// The lanes of one warp this mesh occupies, as a ``__shfl`` mask. Only a
/// mesh inside one warp has one, which is the only mesh that is asked.
template <class TMesh> CUTE_HOST_DEVICE constexpr unsigned lane_mask() {
    constexpr int extent = instances<TMesh>();
    static_assert(extent <= kWarpSize,
                  "ops::sync: a lane mask names the lanes of one warp, and "
                  "this mesh holds more threads than a warp has");
    constexpr int first = tilefoundry::offset(TMesh{}) % kWarpSize;
    return extent == kWarpSize ? 0xffffffffu : ((1u << extent) - 1u) << first;
}

/// Classify a mesh into its synchronization tier.
template <class TMesh> CUTE_HOST_DEVICE constexpr Tier classify() {
    constexpr int first = tilefoundry::offset(TMesh{});
    if constexpr (TMesh::scope == TopologyScope::cta) {
        /// A CTA mesh based at zero covers the grid.
        return first == 0 ? Tier::grid : Tier::illegal_grid_slice;
    } else if constexpr (TMesh::scope == TopologyScope::thread) {
        constexpr int count = instances<TMesh>();
        /// A mesh that is every thread of its level needs no warp shape:
        /// whatever shape it is, one barrier for the block covers it.
        if constexpr (first == 0 && count == int(program_dim<TMesh::scope>()))
            return count <= kWarpSize ? Tier::warp : Tier::block;
        else if constexpr (!tilefoundry::is_warped(TMesh{}))
            return Tier::illegal_warp_layout;
        else if constexpr (count == kWarpSize)
            return Tier::warp;
        else if constexpr (warps_run_together<TMesh>())
            return Tier::named;
        else
            return Tier::illegal_warp_layout;
    } else {
        /// Exhaustive, so that what is not a level cannot be answered as one.
        return Tier::illegal_scope;
    }
}

template <class TMesh> CUTE_HOST_DEVICE constexpr plan_t plan() {
    constexpr Tier tier = classify<TMesh>();
    return {tier,  tilefoundry::offset(TMesh{}), instances<TMesh>(),
            false, tier == Tier::grid,           tier == Tier::named};
}

/// The meshes that name no barrier, refused however they are handed in.
template <Tier tier, class Dep>
CUTE_HOST_DEVICE constexpr void reject_barrierless() {
    if constexpr (tier == Tier::illegal_grid_slice)
        static_assert(dependent_false_v<Dep>,
                      "ops::sync: a partial-grid CTA mesh has no barrier");
    else if constexpr (tier == Tier::illegal_warp_layout)
        static_assert(dependent_false_v<Dep>,
                      "ops::sync: the mesh warp layout must be contiguous");
    else if constexpr (tier == Tier::illegal_scope)
        static_assert(dependent_false_v<Dep>,
                      "ops::sync: a mesh's scope must be cta or thread");
    else
        static_assert(dependent_false_v<Dep>,
                      "ops::sync: barrier tier is missing from dispatch");
}

template <Tier P, class... Ts> CUTE_HOST_DEVICE constexpr void reject() {
    reject_barrierless<P, Ts...>();
}

/// Every CTA of the launch.
struct Grid {
    __device__ void operator()(unsigned int *bar) const {
        if (bar != nullptr)
            grid_barrier(bar);
        else
            cooperative_groups::this_grid().sync();
    }
};

/// Every thread of the block.
struct Block {
    __device__ void operator()() const { __syncthreads(); }
};

/// The threads of one warp or contiguous lane subset.
struct Warp {
    __device__ void operator()(unsigned mask = 0xffffffffu) const {
        __syncwarp(mask);
    }
};

/// A warp-aligned run of ``Count`` threads from ``Base`` on barrier ``BarId``.
template <class TMesh, int Count, int BarId> struct Named {
    __device__ void operator()() const {
        TMesh mesh{};
        if (tilefoundry::contains(mesh, tilefoundry::program_ids()))
            asm volatile("bar.sync %0, %1;" ::"n"(BarId), "n"(Count));
    }
};

}
