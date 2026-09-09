/// CUDA sync op public entry. Included in-context from runtime.cuh inside
/// namespace tilefoundry::ops.
#pragma once

#include "sync/sync_impl.h"

/// One of the fifteen named hardware barriers, as a type.
///
/// A type and not an ``int`` because the id must be a compile-time immediate
/// of ``bar.sync``, refused out of range where it is written. A ``consteval``
/// constructor would let ``sync(mesh, 3)`` do that, but nvcc 13.2 answers an
/// out-of-range argument to one with broken IR instead of a diagnosis.
///
/// Barrier 0 is refused too: ``__syncthreads`` arrives at it, so a subset of
/// the block posting to it releases a whole-block barrier the rest is still
/// walking toward.
template <int Id> struct BarrierId {
    static_assert(Id != 0,
                  "ops::sync: barrier 0 is the one __syncthreads arrives at; a "
                  "run inside the block must take one of 1..15, or it releases "
                  "a whole-block barrier the rest of the block is still "
                  "walking toward");
    static_assert(Id == 0 || (Id >= 1 && Id <= 15),
                  "ops::sync: a named barrier id must be 1..15 -- the hardware "
                  "has sixteen barriers and __syncthreads holds 0. Zero is let "
                  "through this condition so that the one id both assertions "
                  "are about is diagnosed once, by the sentence above that "
                  "explains it");
    static constexpr int value = Id;
};

/// ``bar_id<3>``, so the call site reads as an id and not as a type.
template <int Id> inline constexpr BarrierId<Id> bar_id{};

/// Synchronise every instance of ``mesh``.
///
/// The mesh says *which* barrier; it cannot say what the barrier is *made of*:
///
///     sync(block_mesh)                    // nothing to supply
///     sync(warp_mesh)                     // nothing to supply
///     sync(grid_mesh, tf_grid_bar_state)  // a counter the module owns
///     sync(sub_mesh, bar_id<3>)           // one of the 15 named barriers
///
/// The runtime allocates none: a grid counter belongs to the module, a free
/// named barrier to the whole kernel -- not to one mesh's template.

struct no_resource_t {};
template <class T> struct barrier_id_traits {
    static constexpr bool value = false;
};
template <int Id> struct barrier_id_traits<BarrierId<Id>> {
    static constexpr bool value = true;
    static constexpr int id = Id;
};

/// The grid, which needs the module's counter.
///
/// A null counter is the other grid barrier: a cooperative launch's grid group.
/// Which exists is a fact about the launch, so it stays the caller's to state.
template <class TMesh, TopologyScope... Topos, class TRes = no_resource_t>
__device__ inline void sync(Mesh<TMesh, Topos...> const &mesh,
                            TRes resource = {}) {
    using mesh_t = Mesh<TMesh, Topos...>;
    constexpr auto tier = sync_impl::classify<mesh_t>();
    if constexpr (std::is_same_v<TRes, unsigned int *>) {
        if constexpr (tier == sync_impl::Tier::grid)
            sync_impl::Grid{}(resource);
        else {
            static_assert(dependent_false_v<mesh_t>,
                          "ops::sync: grid counter requires a CTA mesh");
        }
    } else if constexpr (barrier_id_traits<TRes>::value) {
        if constexpr (tier == sync_impl::Tier::named)
            sync_impl::Named<sync_impl::base<mesh_t>(),
                             sync_impl::instances<mesh_t>(),
                             barrier_id_traits<TRes>::id>{}();
        else {
            static_assert(
                dependent_false_v<mesh_t>,
                "ops::sync: named barrier requires a warp-aligned subset");
        }
    } else if constexpr (std::is_integral_v<TRes>) {
        static_assert(dependent_false_v<TRes>,
                      "ops::sync: use bar_id<n>, not an integer barrier id");
    } else if constexpr (std::is_same_v<TRes, no_resource_t>) {
        if constexpr (tier == sync_impl::Tier::warp)
            sync_impl::Warp{}();
        else if constexpr (tier == sync_impl::Tier::block)
            sync_impl::Block{}();
        else if constexpr (tier == sync_impl::Tier::grid)
            static_assert(dependent_false_v<mesh_t>,
                          "ops::sync: a CTA mesh needs the module's "
                          "grid-barrier counter");
        else if constexpr (tier == sync_impl::Tier::named) {
            static_assert(dependent_false_v<mesh_t>,
                          "ops::sync: a warp-aligned run inside the block "
                          "needs a named barrier id");
        } else {
            sync_impl::reject_barrierless<tier, mesh_t>();
        }
    } else {
        static_assert(dependent_false_v<TRes>,
                      "ops::sync: unsupported resource");
    }
}
