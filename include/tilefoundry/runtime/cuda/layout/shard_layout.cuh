// CUDA shard layout surface. Included in-context from runtime.cuh inside
// namespace tilefoundry.
#pragma once

// Topology is parameterised by its scope + total size at compile time.
template <TopologyScope Scope, int Size> struct Topology {
    static constexpr TopologyScope scope = Scope;
    static constexpr int size = Size;
};

// Mesh<topology, cute_layout>: binds a topology to a MeshLayout.
template <class TTopo, class TMeshLayout> struct Mesh {
    using topology = TTopo;
    using layout = TMeshLayout;
    TMeshLayout layout_value;
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

// Legacy aliases in tilefoundry:: for backward compat during migration.
using shard::B;
using shard::Dynamic;
using shard::P;
using shard::S;
