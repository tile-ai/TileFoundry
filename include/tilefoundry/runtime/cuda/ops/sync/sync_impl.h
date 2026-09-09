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
///
/// Scope says the level and the layout how much of it: a thread mesh that fits
/// in a warp needs only the warp's convergence, a larger one the block's, and a
/// CTA mesh the grid's. The mesh already carries all of that, so a call site
/// that names the barrier is restating what it was handed -- which is how the
/// two drift apart when the mesh is reshaped. A mesh that names none is refused
/// where it is used rather than given the next one up: each is a deadlock.
enum class Tier {
    grid,
    block,
    warp,
    named,
    illegal_grid_slice,
    illegal_ragged,
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
/// Off the layout, like ``base`` below, and through the layout system's own
/// ``mesh_instances`` rather than a second copy of the expression.
template <class TMesh> CUTE_HOST_DEVICE constexpr int instances() {
    return tilefoundry::detail::mesh_instances<TMesh, TopologyScope::thread>();
}

/// The first instance a mesh covers, in its topology's own numbering.
///
/// Off the layout, because that is where the IR keeps it: a sliced mesh is a
/// ``ComposedLayout`` whose offset is the slice origin, and an un-sliced one is
/// a plain layout starting at zero.
template <class TMesh> CUTE_HOST_DEVICE constexpr int base() {
    return tilefoundry::detail::mesh_offset<typename TMesh::layout>();
}

/// How many instances the launch gives the level ``TMesh`` names.
///
/// The scope comes through ``TMesh`` and not written out as
/// ``TopologyScope::thread``. Every module specialises ``program_shape``
/// *after* this header; a written-out scope resolves against the primary
/// template where ``program_dim`` is defined -- before the specialisation
/// exists -- and nvcc reports "explicit specialization must precede its first
/// use". Naming it through ``TMesh`` defers the lookup into the kernel.
template <class TMesh> CUTE_HOST_DEVICE constexpr int level_instances() {
    constexpr auto scope = TMesh::first_scope;
    return int(program_dim<scope>());
}

/// The choices ``tir.Sync``'s ``classify`` makes, made the same way, so a
/// hand-written kernel and a generated one cannot disagree about a mesh.
///
/// Each ``Tier`` is one ``SyncBarrier`` of ``ir/tir/sync.py``: grid/GRID,
/// block/SYNCTHREADS, warp/SYNCWARP, named/BAR_SYNC, and the two ``illegal_``
/// tiers for what Python raises ``VerifyError`` on. Three of its checks are
/// not repeated here, because a mesh failing them never reaches codegen: one
/// coordinate per instance, one contiguous interval, and fitting inside the
/// block. ``participation`` raises before a line is emitted.
template <class TMesh> CUTE_HOST_DEVICE constexpr Tier classify() {
    constexpr int first = base<TMesh>();
    if constexpr (tilefoundry::detail::mesh_names_level<TMesh,
                                                        TopologyScope::cta>()) {
        /// A CTA mesh is never asked its size: a launch-sized grid has no
        /// static one, and the grid barrier counts CTAs out of ``gridDim``.
        /// Based past zero it covers only some of the grid, and the CTAs
        /// outside it never arrive.
        return first == 0 ? Tier::grid : Tier::illegal_grid_slice;
    } else if constexpr (tilefoundry::detail::mesh_names_level<
                             TMesh, TopologyScope::thread>()) {
        constexpr int count = instances<TMesh>();
        constexpr int block = level_instances<TMesh>();
        using warp_view = tilefoundry::detail::MeshWarpView<TMesh>;
        constexpr bool single_warp = warp_view::warps() == 1;
        /// "Covers the whole block" is ``full_cta`` -- based at zero *and* as
        /// wide as the block -- not "based at zero", which is what this asked.
        /// They differ on a mesh narrowed from the front: ``m[0:2,:]`` of a
        /// 128-thread block is 64 threads at offset 0, a subset to Python and
        /// the whole block to a base test, so one emitted a named barrier and
        /// the other ``__syncthreads`` -- and 64 threads arriving where the
        /// other 64 never do is a hang. The width comes from the launch, so it
        /// cannot drift from it.
        if constexpr (first == 0 && count == block) {
            return warp_view::whole_warps && warp_view::warps() == 1
                       ? Tier::warp
                       : Tier::block;
        } else if constexpr (single_warp) {
            return Tier::warp;
        } else if constexpr (warp_view::whole_warps &&
                             warp_view::warps_contiguous()) {
            /// A warp-aligned run is what the named barrier counts; anything
            /// else cuts a warp in half, part of it inside the barrier and
            /// part outside.
            return Tier::named;
        } else {
            return Tier::illegal_ragged;
        }
    } else {
        /// Exhaustive, so that what is not a level cannot be answered as one.
        /// ``TopologyScope`` also enumerates ``warp``, which has no
        /// ``program_id`` to name a participant with, and ``scope_count``,
        /// which is a sentinel; a trailing ``else`` standing for ``thread``
        /// handed both of them the block's barrier.
        return Tier::illegal_scope;
    }
}

template <class TMesh> CUTE_HOST_DEVICE constexpr plan_t plan() {
    constexpr Tier tier = classify<TMesh>();
    return {tier,  base<TMesh>(),      instances<TMesh>(),
            false, tier == Tier::grid, tier == Tier::named};
}

/// The meshes that name no barrier, refused however they are handed in.
///
/// In its own function because every ``sync`` overload has to make the same
/// refusal: a mesh that names no barrier is a deadlock whether or not the
/// caller also handed over a resource, and one overload forgetting to say so
/// would make the diagnosis depend on which resource was supplied.
///
/// Reached only from an overload's own tail, once that overload has dispatched
/// or refused every tier that *does* name a barrier -- so its own tail stands
/// for a ``Tier`` nobody has handled, and says so instead of returning.
template <Tier tier, class Dep>
CUTE_HOST_DEVICE constexpr void reject_barrierless() {
    if constexpr (tier == Tier::illegal_grid_slice)
        static_assert(dependent_false_v<Dep>,
                      "ops::sync: a CTA mesh that covers only part of the grid "
                      "has no barrier -- the CTAs outside it never arrive");
    else if constexpr (tier == Tier::illegal_ragged)
        static_assert(dependent_false_v<Dep>,
                      "ops::sync: a cross-warp subset must be warp-aligned -- "
                      "both the base and the count must be multiples of 32");
    else if constexpr (tier == Tier::illegal_scope)
        static_assert(dependent_false_v<Dep>,
                      "ops::sync: a mesh's scope must be cta or thread -- the "
                      "warp level has no program_id to name its participants "
                      "with and scope_count is a sentinel, so a warp-sized "
                      "grouping is an axis of a thread mesh's layout");
    else
        static_assert(dependent_false_v<Dep>,
                      "ops::sync: this Tier names a barrier and reached the "
                      "tail that stands for the ones that do not -- an "
                      "overload's dispatch chain is missing a case");
}

template <Tier P, class... Ts> CUTE_HOST_DEVICE constexpr void reject() {
    reject_barrierless<P, Ts...>();
}

/// Every CTA of the launch.
///
/// Two ways to make every CTA wait, and which is available is a property of the
/// launch rather than of the mesh: a module that codegen gave a barrier counter
/// spins on it, and a cooperative launch has the grid group instead. The
/// pointer is the discriminator because it is the thing only the caller knows.
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

/// The threads of one warp.
///
/// The full mask, even for a mesh narrower than a warp. ``sync`` gates nothing:
/// every thread of the block runs the statement, so the whole warp is at it and
/// a mask naming only the mesh's lanes would describe a convergence that is not
/// the one happening. The narrower masks the emitter used to compute went with
/// a call site that had already branched.
struct Warp {
    __device__ void operator()() const { __syncwarp(); }
};

/// A warp-aligned run of ``Count`` threads from ``Base`` on barrier ``BarId``.
///
/// The id is handed in, not derived. It used to be ``1 + (Base / 32 - 1) %
/// 15``, which reads like it keeps two runs apart and does not: the hardware
/// counts arrivals *per barrier id*, so two meshes sharing a base and
/// differing in count -- ``m[64:96]`` and ``m[64:128]`` -- got the same id and
/// the first to fill its count released the other's threads. Who owns which
/// id is a fact about the whole kernel, the caller's to know and not this
/// template's.
template <int Base, int Count, int BarId> struct Named {
    __device__ void operator()() const {
        const int tid =
            int(tilefoundry::program_id<tilefoundry::TopologyScope::thread>());
        if (tid >= Base && tid < Base + Count)
            asm volatile("bar.sync %0, %1;" ::"n"(BarId), "n"(Count));
    }
};

}
