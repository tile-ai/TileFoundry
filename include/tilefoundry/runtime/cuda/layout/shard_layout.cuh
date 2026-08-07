// CUDA shard layout surface. Included in-context from runtime.cuh inside
// namespace tilefoundry.
#pragma once

// Topology is parameterised by its scope + total size at compile time.
template <TopologyScope Scope, int Size> struct Topology {
    static constexpr TopologyScope scope = Scope;
    static constexpr int size = Size;
};

// Mesh<topology, cute_layout, more_topologies...>: binds one or more
// topologies to a MeshLayout. The topology product is the mesh domain, which
// the MeshLayout then subdivides into logical axes; ``topology`` stays the
// primary (coarsest) one so single-topology users are unaffected.
template <class TTopo, class TMeshLayout, class... TMoreTopos> struct Mesh {
    using topology = TTopo;
    using layout = TMeshLayout;
    TMeshLayout layout_value;

    // Linearized position of the current execution instance within the mesh
    // domain, coarsest topology outermost. This is the index the MeshLayout is
    // built against: a cta x thread mesh addresses as
    // ``cta_id * thread_size + thread_id``. With no extra topologies this is
    // just ``program_id<TTopo::scope>()``.
    CUTE_HOST_DEVICE static size_t linear_id() noexcept {
        size_t id = program_id<TTopo::scope>();
        ((id = id * size_t(TMoreTopos::size) + program_id<TMoreTopos::scope>()),
         ...);
        return id;
    }
};

// ShardLayout<layout, attrs_tuple, mesh>: spec 003 shard layout surface.
template <class TLayout, class TAttrs, class TMesh> struct ShardLayout {
    using layout = TLayout;
    using attrs = TAttrs;
    using mesh = TMesh;
    TLayout layout_value;
    TMesh mesh_value;
};

// Per-axis shard attributes.
namespace shard {
template <int Axis> struct S {
    static constexpr int axis = Axis;
};
struct B {};
template <class Reduction> struct P {
    using reduction = Reduction;
};
struct Dynamic {};
} // namespace shard
