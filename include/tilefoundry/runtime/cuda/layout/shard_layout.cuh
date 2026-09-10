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

/// The mesh shape a ShardLayout's attrs are indexed against: its axes, flat.
///
/// A mesh naming several levels states them grouped one nest per level, but
/// an attr answers for one axis whichever level owns it, so the attrs run
/// against the axes with the grouping taken back out.
template <class SL>
using shard_mesh_shape_t = cute::remove_cvref_t<decltype(cute::flatten(
    cute::shape(typename SL::mesh::layout_type{})))>;

/// The mesh shape as the mesh states it: one entry per level when it names
/// several, and one per axis when it names one. What a coord is shaped like.
template <class SL>
using shard_mesh_groups_t = cute::remove_cvref_t<decltype(cute::shape(
    typename SL::mesh::layout_type{}))>;

/// How many mesh axes one entry of that grouped shape holds.
template <class Entry> CUTE_HOST_DEVICE constexpr int mesh_entry_rank(Entry) {
    if constexpr (cute::is_integral<cute::remove_cvref_t<Entry>>::value)
        return 1;
    else
        return int(cute::rank(cute::remove_cvref_t<Entry>{}));
}

/// Where entry ``E`` starts in the flat run of mesh axes the attrs index.
template <class Groups, size_t E, size_t... Is>
CUTE_HOST_DEVICE constexpr int mesh_entry_base(std::index_sequence<Is...>) {
    return (0 + ... + (Is < E ? mesh_entry_rank(cute::get<Is>(Groups{})) : 0));
}

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
        cute::tuple_size<shard_mesh_shape_t<SL>>::value>;
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
                 int(cute::size(cute::get<Is>(shard_mesh_shape_t<SL>{}))))
          : void()),
     ...);
    return divisor;
}

template <class SL, int Axis> CUTE_HOST_DEVICE constexpr int shard_divisor() {
    return shard_divisor_over<SL, Axis>(
        std::make_index_sequence<
            cute::tuple_size<shard_mesh_shape_t<SL>>::value>{});
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
                 int(cute::size(cute::get<Is>(shard_mesh_shape_t<SL>{}))))
          : void()),
     ...);
    return inner;
}

template <class SL, size_t Ax, int Axis>
CUTE_HOST_DEVICE constexpr int shard_inner() {
    return shard_inner_over<SL, Ax, Axis>(
        std::make_index_sequence<
            cute::tuple_size<shard_mesh_shape_t<SL>>::value>{});
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

/// Tensor axis ``I``'s local stride: the shard layout's own, splitting or not.
template <size_t I, class L, class A, class M>
CUTE_HOST_DEVICE constexpr auto local_stride(ShardLayout<L, A, M> const &sl) {
    return cute::get<I>(cute::stride(sl.layout_value));
}

/// What one step along mesh axis ``Ax`` costs the storage offset. Broadcast
/// and Partial leave the tensor whole, so a step along them costs nothing.
template <size_t Ax, class L, class A, class M>
CUTE_HOST_DEVICE constexpr auto
mesh_axis_stride(ShardLayout<L, A, M> const &sl) {
    using attr_t = cute::remove_cvref_t<decltype(cute::get<Ax>(A{}))>;
    constexpr int k = attr_axis<attr_t>::value;
    if constexpr (k >= 0) {
        constexpr int inner = shard_inner<ShardLayout<L, A, M>, Ax, k>();
        return local_extent<size_t(k)>(sl) * cute::Int<inner>{} *
               local_stride<size_t(k)>(sl);
    } else {
        static_assert(attr_leaves_tensor_whole<attr_t>(),
                      "shard layout: this attr must leave the tensor whole");
        return cute::Int<0>{};
    }
}

/// One entry of the mesh shape's strides, shaped the way that entry is.
///
/// The attrs are a flat run over the mesh's axes and the shape may group
/// them by level, so an entry takes the run from where it starts for as many
/// axes as it holds. A level's nest gets a nest of strides; a bare axis gets
/// the one stride it is.
template <size_t E, class L, class A, class M>
CUTE_HOST_DEVICE constexpr auto entry_stride(ShardLayout<L, A, M> const &sl) {
    using groups_t = shard_mesh_groups_t<ShardLayout<L, A, M>>;
    using entry_t = cute::remove_cvref_t<decltype(cute::get<E>(groups_t{}))>;
    constexpr int base = mesh_entry_base<groups_t, E>(
        std::make_index_sequence<cute::tuple_size<groups_t>::value>{});
    if constexpr (cute::is_integral<entry_t>::value) {
        return mesh_axis_stride<size_t(base)>(sl);
    } else {
        return [&]<size_t... J>(std::index_sequence<J...>) {
            return cute::make_stride(mesh_axis_stride<size_t(base) + J>(sl)...);
        }(std::make_index_sequence<cute::tuple_size<entry_t>::value>{});
    }
}

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
                                 cute::make_stride(local_stride<Is>(sl)...));
    }(std::make_index_sequence<t_rank>{});
}

/// What ``cute::slice_and_offset`` is to a Layout, this is to a ShardLayout:
/// the instance at ``coord`` holds this layout, beginning this many elements
/// into the whole tensor's engine.
///
/// The offset is ``crd2idx`` against a layout over the mesh -- its own shape,
/// grouping and all, so ``mesh_coords`` indexes it as it stands -- whose
/// strides say how far along the tensor one step of each mesh axis moves the
/// slice's beginning.
template <class L, class A, class M, class Coord>
CUTE_HOST_DEVICE constexpr auto
local_layout_and_offset(ShardLayout<L, A, M> const &sl, Coord const &coord) {
    using SL = ShardLayout<L, A, M>;
    static_assert(
        shard_attrs_match_mesh<SL>(),
        "one attr per mesh axis: this layout has one stride per attr");
    return [&]<size_t... E>(std::index_sequence<E...>) {
        return cute::make_tuple(
            local_layout(sl),
            int(cute::crd2idx(coord, cute::shape(sl.mesh_value.layout),
                              cute::make_stride(entry_stride<E>(sl)...))));
    }(std::make_index_sequence<
               cute::tuple_size<shard_mesh_groups_t<SL>>::value>{});
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
