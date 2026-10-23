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

/// The mesh shape as the mesh states it: one entry per level when it names
/// several, and one per axis when it names one. What a coord is shaped like.
template <class SL>
using shard_mesh_shape_t = cute::remove_cvref_t<decltype(cute::shape(
    typename SL::mesh::layout_type{}))>;

/// The same shape with the grouping taken back out, which is what the attrs
/// are indexed against: an attr answers for one axis whichever level owns it.
template <class SL>
using shard_mesh_flat_t = cute::remove_cvref_t<decltype(cute::flatten(
    cute::shape(typename SL::mesh::layout_type{})))>;

/// The mesh axis that splits tensor axis ``Axis``, or -1 when none does.
template <class SL, int Axis, size_t... Is>
CUTE_HOST_DEVICE constexpr int
shard_mesh_axis_over(std::index_sequence<Is...>) {
    using attrs_t = typename SL::attrs;
    int found = -1;
    ((attr_axis<cute::remove_cvref_t<decltype(cute::get<Is>(attrs_t{}))>>::
                  value == Axis
          ? void(found = int(Is))
          : void()),
     ...);
    return found;
}

/// Count mesh axes that name tensor axis Axis.
template <class SL, int Axis, size_t... Is>
CUTE_HOST_DEVICE constexpr int
shard_mesh_axes_over(std::index_sequence<Is...>) {
    using attrs_t = typename SL::attrs;
    return (0 + ... +
            int(attr_axis<cute::remove_cvref_t<decltype(cute::get<Is>(
                    attrs_t{}))>>::value == Axis));
}

/// The mesh axis that names tensor axis ``Axis``, and ``-1`` where none does.
template <class SL, int Axis> CUTE_HOST_DEVICE constexpr int shard_mesh_axis() {
    using seq = std::make_index_sequence<
        cute::tuple_size<shard_mesh_flat_t<SL>>::value>;
    return shard_mesh_axis_over<SL, Axis>(seq{});
}

/// What tensor axis ``Axis`` is divided by: every mesh axis cutting it.
///
/// Several may. Two axes of one level cut a tensor axis into a grid, and so
/// do two levels, one taking a block of what the other left. Each division is
/// of what the previous one left, so their extents multiply.
template <class SL, int Axis, size_t... Is>
CUTE_HOST_DEVICE constexpr int shard_divisor_over(std::index_sequence<Is...>) {
    using attrs_t = typename SL::attrs;
    int divisor = 1;
    ((attr_axis<cute::remove_cvref_t<decltype(cute::get<Is>(attrs_t{}))>>::
                  value == Axis
          ? void(divisor *=
                 int(cute::size(cute::get<Is>(shard_mesh_flat_t<SL>{}))))
          : void()),
     ...);
    return divisor;
}

template <class SL, int Axis> CUTE_HOST_DEVICE constexpr int shard_divisor() {
    return shard_divisor_over<SL, Axis>(
        std::make_index_sequence<
            cute::tuple_size<shard_mesh_flat_t<SL>>::value>{});
}

/// How much of tensor axis ``Axis`` one step of mesh axis ``Ax`` steps over.
///
/// The axes cutting one tensor axis are ordered outermost first, so a step of
/// one of them clears everything the axes inside it hold.
template <class SL, size_t Ax, int Axis, size_t... Is>
CUTE_HOST_DEVICE constexpr int shard_inner_over(std::index_sequence<Is...>) {
    using attrs_t = typename SL::attrs;
    int inner = 1;
    ((Is > Ax && attr_axis<cute::remove_cvref_t<decltype(cute::get<Is>(
                         attrs_t{}))>>::value == Axis
          ? void(inner *=
                 int(cute::size(cute::get<Is>(shard_mesh_flat_t<SL>{}))))
          : void()),
     ...);
    return inner;
}

template <class SL, size_t Ax, int Axis>
CUTE_HOST_DEVICE constexpr int shard_inner() {
    return shard_inner_over<SL, Ax, Axis>(
        std::make_index_sequence<
            cute::tuple_size<shard_mesh_flat_t<SL>>::value>{});
}

/// The part of a layout that carries strides.
///
/// A swizzled layout is a ``cute::ComposedLayout`` whose first component is a
/// function rather than a stride rule, so CuTe deletes ``stride()`` on it
/// (layout_composed.hpp). Every reader of a step reads it off the layout
/// underneath; the function is put back when the tensor is projected.
template <class L>
CUTE_HOST_DEVICE constexpr L const &affine_portion(L const &layout) {
    return layout;
}

template <class A, class O, class B>
CUTE_HOST_DEVICE constexpr auto
affine_portion(cute::ComposedLayout<A, O, B> const &layout) {
    return layout.layout_b();
}

/// Local extent of tensor axis I.
template <size_t I, class L, class A, class M>
CUTE_HOST_DEVICE constexpr auto local_extent(ShardLayout<L, A, M> const &sl) {
    auto const ext = cute::get<I>(cute::shape(sl.layout_value));
    constexpr int divisor = shard_divisor<ShardLayout<L, A, M>, int(I)>();
    if constexpr (divisor == 1)
        return ext;
    else
        return ext / cute::Int<divisor>{};
}

}

/// Mesh axis ``Ax``'s stride, as ``cute::stride`` is a layout's: what one
/// step along it costs the slice's origin. Broadcast and Partial leave the
/// tensor whole, so a step along them costs nothing.
template <size_t Ax, class L, class A, class M>
CUTE_HOST_DEVICE constexpr auto stride(ShardLayout<L, A, M> const &sl) {
    using attr_t = cute::remove_cvref_t<decltype(cute::get<Ax>(A{}))>;
    constexpr int k = detail::attr_axis<attr_t>::value;
    if constexpr (k >= 0) {
        constexpr int inner =
            detail::shard_inner<ShardLayout<L, A, M>, Ax, k>();
        return detail::local_extent<size_t(k)>(sl) * cute::Int<inner>{} *
               cute::stride<k>(detail::affine_portion(sl.layout_value));
    } else {
        static_assert(detail::attr_leaves_tensor_whole<attr_t>(),
                      "shard layout: this attr must leave the tensor whole");
        return cute::Int<0>{};
    }
}

namespace detail {

/// Whether a shard layout says one thing about each of its mesh's axes.
///
/// Against the mesh's axes flattened, not its modes: a mesh naming several
/// levels states them grouped one nest per level, and its rank is then how
/// many levels it names rather than how many axes the attrs answer for.
template <class SL> CUTE_HOST_DEVICE constexpr bool shard_attrs_match_mesh() {
    return int(cute::tuple_size<typename SL::attrs>::value) ==
           int(cute::rank(
               cute::flatten(cute::shape(typename SL::mesh::layout_type{}))));
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

/// The layout one mesh instance holds: each tensor axis narrowed by what
/// cuts it, keeping the whole tensor's own strides. Every instance holds
/// the same shape, so this does not depend on which instance it is.
template <class L, class A, class M>
CUTE_HOST_DEVICE constexpr auto local_layout(ShardLayout<L, A, M> const &sl) {
    constexpr int t_rank =
        cute::tuple_size<cute::remove_cvref_t<decltype(cute::shape(
            typename ShardLayout<L, A, M>::layout{}))>>::value;
    return [&]<size_t... Is>(std::index_sequence<Is...>) {
        return cute::make_layout(cute::make_shape(local_extent<Is>(sl)...),
                                 cute::make_stride(cute::stride<int(Is)>(
                                     affine_portion(sl.layout_value))...));
    }(std::make_index_sequence<t_rank>{});
}

/// What ``cute::slice_and_offset`` is to a Layout, this is to a ShardLayout:
/// the instance at ``coord`` holds this layout, beginning this many elements
/// into the whole tensor's engine.
///
/// The offset is a layout over the mesh applied to ``coord``: the attrs give
/// one stride per mesh axis, flat, and ``unflatten`` regroups them to the
/// mesh's own shape so ``mesh_coords`` indexes them as it stands.
template <class L, class A, class M, class Coord>
CUTE_HOST_DEVICE constexpr auto
local_layout_and_offset(ShardLayout<L, A, M> const &sl, Coord const &coord) {
    using SL = ShardLayout<L, A, M>;
    static_assert(
        shard_attrs_match_mesh<SL>(),
        "one attr per mesh axis: this layout has one stride per attr");
    return [&]<size_t... Ax>(std::index_sequence<Ax...>) {
        auto mesh_shape = cute::shape(sl.mesh_value.layout);
        auto flat = cute::make_stride(tilefoundry::stride<Ax>(sl)...);
        auto mesh_layout =
            cute::make_layout(mesh_shape, cute::unflatten(flat, mesh_shape));
        return cute::make_tuple(local_layout(sl), int(mesh_layout(coord)));
    }(std::make_index_sequence<
               cute::tuple_size<shard_mesh_flat_t<SL>>::value>{});
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
