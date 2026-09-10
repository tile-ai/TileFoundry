/// CUDA shard layout surface: a tensor layout, one attr per mesh axis, and
/// the mesh those attrs are indexed against. The mesh itself is
/// layout/mesh.cuh.
#pragma once

/// ShardLayout<layout, attrs_tuple, mesh>: spec 003 shard layout surface.
template <class TLayout, class TAttrs, class TMesh> struct ShardLayout {
    using layout = TLayout;
    using attrs = TAttrs;
    using mesh = TMesh;
    TLayout layout_value;
    TMesh mesh_value;
};

/// Per-axis shard attributes.
namespace shard {
template <int Axis> struct S {
    static constexpr int axis = Axis;
};
struct B {};
template <class Reduction> struct P {
    using reduction = Reduction;
};
struct Dynamic {};
}

namespace detail {

/// Whether a shard attribute names a tensor axis (``S<n>``, not ``B``).
template <class A, class = void> struct attr_axis {
    static constexpr int value = -1;
};
template <class A> struct attr_axis<A, std::void_t<decltype(A::axis)>> {
    static constexpr int value = int(A::axis);
};

/// Identify split and partial shard attributes.
template <class A> struct is_split_attr : std::false_type {};
template <int Axis> struct is_split_attr<shard::S<Axis>> : std::true_type {};
template <class A>
inline constexpr bool is_split_attr_v = is_split_attr<A>::value;

template <class A> struct is_partial_attr : std::false_type {};
template <class R> struct is_partial_attr<shard::P<R>> : std::true_type {};
template <class A>
inline constexpr bool is_partial_attr_v = is_partial_attr<A>::value;

/// Whether a mesh-axis attr leaves each instance the whole tensor.
template <class A> CUTE_HOST_DEVICE constexpr bool attr_leaves_tensor_whole() {
    if constexpr (is_split_attr_v<A>) {
        return false;
    } else if constexpr (std::is_same_v<A, shard::B> || is_partial_attr_v<A>) {
        /// Broadcast and Partial do not move the slice origin.
        return true;
    } else {
        static_assert(dependent_false_v<A>,
                      "shard layout: this attr neither names a tensor axis "
                      "nor leaves the tensor whole");
        return true;
    }
}

/// Whether a shard layout says one thing about each of its mesh's axes.
template <class SL> CUTE_HOST_DEVICE constexpr bool shard_attrs_match_mesh() {
    return int(cute::tuple_size<typename SL::attrs>::value) ==
           decltype(cute::rank(typename SL::mesh::layout_type{}))::value;
}

/// A shard layout no mesh axis splits: every instance holds the whole tensor.
template <class SL>
CUTE_HOST_DEVICE constexpr bool shard_layout_is_full_broadcast() {
    static_assert(shard_attrs_match_mesh<SL>(),
                  "one attr per mesh axis: a shard layout must say what each "
                  "of its mesh's axes does with the tensor");
    using attrs_t = typename SL::attrs;
    return [&]<size_t... Is>(std::index_sequence<Is...>) {
        return (
            true && ... &&
            attr_leaves_tensor_whole<
                cute::remove_cvref_t<decltype(cute::get<Is>(attrs_t{}))>>());
    }(std::make_index_sequence<cute::tuple_size<attrs_t>::value>{});
}

}

/// Build a ShardLayout from canonical pieces.

/// The one rule both forms impose, stated where the layout is built.
template <class SL> CUTE_HOST_DEVICE constexpr void check_shard_layout() {
    static_assert(detail::shard_attrs_match_mesh<SL>(),
                  "make_shard_layout: one attr per mesh axis");
}

/// A shape, with row-major strides.
template <class Shape, class TMesh, class Attrs,
          __CUTE_REQUIRES(!cute::is_layout<Shape>::value)>
CUTE_HOST_DEVICE constexpr auto
make_shard_layout(Shape const &shape, TMesh const &mesh, Attrs const &) {
    auto layout = cute::make_layout(shape, cute::GenRowMajor{});
    using sl_t = ShardLayout<decltype(layout), Attrs, TMesh>;
    check_shard_layout<sl_t>();
    return sl_t{layout, mesh};
}

/// A layout the caller already built.
template <class TLayout, class TMesh, class Attrs,
          __CUTE_REQUIRES(cute::is_layout<TLayout>::value)>
CUTE_HOST_DEVICE constexpr auto
make_shard_layout(TLayout const &layout, TMesh const &mesh, Attrs const &) {
    using sl_t = ShardLayout<TLayout, Attrs, TMesh>;
    check_shard_layout<sl_t>();
    return sl_t{layout, mesh};
}
