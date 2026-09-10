/// CUDA sync op public entry. Included in-context from runtime.cuh inside
/// namespace tilefoundry::ops.
#pragma once

#include "sync/sync_impl.h"

/// One of the fifteen named hardware barriers, as a type.
template <int Id> struct BarrierId {
    static_assert(Id != 0,
                  "ops::sync: barrier 0 is reserved for __syncthreads");
    static_assert(Id == 0 || (Id >= 1 && Id <= 15),
                  "ops::sync: a named barrier id must be 1..15");
    static constexpr int value = Id;
};

/// ``bar_id<3>``, so the call site reads as an id and not as a type.
template <int Id> inline constexpr BarrierId<Id> bar_id{};

/// Synchronise every instance of ``mesh``.

struct no_resource_t {};
template <class T> struct barrier_id_traits {
    static constexpr bool value = false;
};
template <int Id> struct barrier_id_traits<BarrierId<Id>> {
    static constexpr bool value = true;
    static constexpr int id = Id;
};

/// The grid, which needs the module's counter.
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
            sync_impl::Named<mesh_t, sync_impl::instances<mesh_t>(),
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
            sync_impl::Warp{}(sync_impl::lane_mask<mesh_t>());
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
