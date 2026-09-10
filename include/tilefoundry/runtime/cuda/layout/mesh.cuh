/// A mesh: the hardware levels it names, and one CuTe layout whose values are
/// those levels' ids. The C++ half of [shard §5](docs/spec/shard.md#5-mesh).
///
/// The struct carries the two fields and nothing else. Everything asked of a
/// mesh is either a CuTe question about that layout -- ``cute::size`` /
/// ``shape`` / ``rank`` / evaluation, which read through a ``ComposedLayout``
/// slice on their own -- or one of the free functions below.
#pragma once

namespace detail {
/// The pack's own first value. Reading it off ``decltype(tuple){}`` instead
/// hands back the zeroth enumerator, which is a different level.
template <TopologyScope T0, TopologyScope...> struct first_scope {
    static constexpr TopologyScope value = T0;
};
}

/// The levels, and the layout whose values are their ids. How many instances
/// a level has is the launch's, not the mesh's: one .cu is one launch, so
/// ``program_dim<scope>()`` states it and the same mesh in another
/// translation unit is another launch's mesh.
template <class TLayout, TopologyScope... Topos> struct Mesh {
    using layout_type = TLayout;
    static constexpr int level_count = sizeof...(Topos);
    static constexpr TopologyScope scope = detail::first_scope<Topos...>::value;
    TLayout layout;
};

namespace detail {

/// Where ``S`` sits in the pack, and ``sizeof...`` when it names no level.
template <TopologyScope S, TopologyScope... Ts>
CUTE_HOST_DEVICE constexpr int level_index() {
    constexpr bool hit[] = {(Ts == S)...};
    for (int i = 0; i < int(sizeof...(Ts)); ++i)
        if (hit[i])
            return i;
    return int(sizeof...(Ts));
}

/// A mesh whose axes span more than one level has no rule assigning them.
template <class TMesh> CUTE_HOST_DEVICE constexpr void one_level() {
    static_assert(TMesh::level_count == 1,
                  "Mesh: multiple topology levels have ids, but no rule "
                  "assigns mesh axes to those levels");
}

/// The plain layout under a slice: what carries ``get_hier_coord``.
template <class L> CUTE_HOST_DEVICE constexpr auto positions_of(L const &l) {
    if constexpr (cute::is_composed_layout<cute::remove_cvref_t<L>>::value)
        return l.layout_b();
    else
        return l;
}

/// The mesh's id axes, in the order CuTe's algebra reads them.
///
/// ``reverse`` because a mesh is row-major and CuTe is not -- the reason is
/// on ``reverse`` itself in layout/cute_ext.cuh -- and ``filter`` because an
/// axis of one position, which CuTe writes as stride zero, names no id.
template <class L> CUTE_HOST_DEVICE constexpr auto id_axes(L const &l) {
    return cute::filter(reverse(cute::flatten(positions_of(l))));
}

}

/// The level's own mesh, out of a mesh that may name several.
///
/// A mesh naming one level is that level's, whole. A mesh naming more states
/// its axes grouped one nest per level, each already in that level's own
/// numbering, and this is ``cute::get`` of the nest -- which is why it shares
/// the name. The composite is a grouping, not a map: only the nests are
/// evaluated, so a mesh naming several levels is asked one at a time.
template <TopologyScope S, class L, TopologyScope... Topos>
CUTE_HOST_DEVICE constexpr auto get(Mesh<L, Topos...> const &mesh) {
    constexpr int at = detail::level_index<S, Topos...>();
    static_assert(at < int(sizeof...(Topos)),
                  "get<level>: this mesh does not name that level");
    if constexpr (sizeof...(Topos) == 1) {
        return mesh;
    } else {
        static_assert(!cute::is_composed_layout<cute::remove_cvref_t<L>>::value,
                      "get<level>: a mesh naming several levels cannot also be "
                      "sliced");
        auto const level = cute::get<at>(mesh.layout);
        return Mesh<cute::remove_cvref_t<decltype(level)>, S>{level};
    }
}

/// The first id the mesh covers, in its level's own numbering: a slice's
/// ``cute::ComposedLayout::offset()``, and zero for a whole level.
template <class L, TopologyScope... Topos>
CUTE_HOST_DEVICE constexpr int offset(Mesh<L, Topos...> const &mesh) {
    if constexpr (cute::is_composed_layout<cute::remove_cvref_t<L>>::value) {
        using inner = cute::remove_cvref_t<decltype(mesh.layout.layout_a())>;
        using off = cute::remove_cvref_t<decltype(mesh.layout.offset())>;
        static_assert(std::is_same_v<inner, cute::identity>,
                      "offset: a mesh layout's first component must be "
                      "cute::identity");
        static_assert(cute::is_static<off>::value,
                      "offset: a mesh layout's offset must be a compile-time "
                      "number");
        return int(off{});
    } else {
        return 0;
    }
}

/// Whether ``coord`` names an instance of ``mesh``.
///
/// The coord is one id per topology level, which is what ``program_ids()``
/// hands back and what a mesh grouped one nest per level is indexed by.
template <class L, TopologyScope... Topos, class Coord>
CUTE_HOST_DEVICE constexpr bool contains(Mesh<L, Topos...> const &mesh,
                                         Coord const &coord) {
    using mesh_t = Mesh<L, Topos...>;
    detail::one_level<mesh_t>();
    const int rel = int(cute::get<int(mesh_t::scope)>(coord)) - offset(mesh);
    if (rel < 0)
        return false;
    auto const positions = detail::positions_of(mesh.layout);
    auto const at = positions.get_hier_coord(rel);
    /// idx2crd answers for a hole and for overshoot too; only the value it was
    /// built to reproduce says the id is one this mesh actually covers.
    return cute::elem_less(at, cute::shape(positions)) &&
           int(positions(at)) == rel;
}

/// Which instance ``coord`` acts as, as a single number -- CuTe's own
/// ``Layout::get_1d_coord``, so a layout over the same shape takes it.
///
/// Not the same question as ``contains``, and on a mesh narrower than its
/// level not the same answer: ``idx2crd`` is ``(id / stride) % extent``, so
/// thread 64 of a 128-thread block acts as instance 0 of a 32-instance mesh
/// and the upper warps repeat what the lowest warp does. A slice does not
/// repeat -- its body runs under ``contains`` -- so a coord from outside
/// one reaching here is a codegen fault, not an instance to fold.
template <class L, TopologyScope... Topos, class Coord>
CUTE_HOST_DEVICE constexpr int get_1d_coord(Mesh<L, Topos...> const &mesh,
                                            Coord const &coord) {
    using mesh_t = Mesh<L, Topos...>;
    detail::one_level<mesh_t>();
    if constexpr (cute::is_composed_layout<cute::remove_cvref_t<L>>::value)
        assert(contains(mesh, coord));
    const int rel = int(cute::get<int(mesh_t::scope)>(coord)) - offset(mesh);
    auto const positions = detail::positions_of(mesh.layout);
    return int(
        cute::crd2idx(positions.get_hier_coord(rel), cute::shape(positions)));
}

/// Whether the mesh's ids say where the warps are.
///
/// The axes arrive in CuTe's order, so mode zero is the fastest and its
/// extent is how many lanes run together: ``(32,..):(1,..)`` is one warp and
/// the next axis carries the warps, ``(128,..):(1,..)`` holds four that
/// ``as_warped`` splits out, and ``(16,..):(1,..)`` is half a warp shared
/// between instances. The third reaches neither ``bar.sync``, which counts
/// whole warps, nor one ``__syncwarp``, so it is refused by name.
template <class L, TopologyScope... Topos>
CUTE_HOST_DEVICE constexpr bool is_warped(Mesh<L, Topos...> const &mesh) {
    using ids_t = cute::remove_cvref_t<decltype(detail::id_axes(mesh.layout))>;
    constexpr int r = decltype(cute::rank(ids_t{}))::value;
    if constexpr (r == 0) {
        return false;
    } else if constexpr (int(cute::get<0>(cute::stride(ids_t{}))) != 1 ||
                         int(cute::get<0>(cute::shape(ids_t{}))) % kWarpSize !=
                             0) {
        return false;
    } else {
        return [&]<size_t... Is>(std::index_sequence<Is...>) {
            return (
                (int(cute::get<Is + 1>(cute::stride(ids_t{}))) % kWarpSize ==
                 0) &&
                ...);
        }(std::make_index_sequence<r - 1>{});
    }
}

/// The same mesh with its lanes stated as an axis of their own: the fastest
/// axis of ``E`` positions becomes ``(E/32, 32)``, so the last axis is one
/// warp and the axis before it steps between warps.
///
/// Idempotent -- a fastest axis of exactly 32 is already that -- and every
/// warp question is then a shape or a stride of the result.
template <class L, TopologyScope... Topos>
CUTE_HOST_DEVICE constexpr auto as_warped(Mesh<L, Topos...> const &mesh) {
    static_assert(is_warped(Mesh<L, Topos...>{}),
                  "as_warped: this mesh's fastest axis must step by one "
                  "and run whole warps");
    using ids_t = cute::remove_cvref_t<decltype(detail::id_axes(mesh.layout))>;
    /// Mode zero is the fastest here, so a warp-sized tile cuts the axis a
    /// warp runs along; the result turns back into the mesh's own order.
    if constexpr (int(cute::get<0>(cute::shape(ids_t{}))) == kWarpSize) {
        auto rowwise = reverse(ids_t{});
        return Mesh<cute::remove_cvref_t<decltype(rowwise)>, Topos...>{rowwise};
    } else {
        auto cut = cute::flatten(
            cute::logical_divide(ids_t{}, cute::Int<kWarpSize>{}));
        auto rowwise = reverse(cut);
        return Mesh<cute::remove_cvref_t<decltype(rowwise)>, Topos...>{rowwise};
    }
}

/// A mesh over ``extents``, row-major, as the Python ``make_mesh`` builds one.
template <TopologyScope Scope, class Extents>
CUTE_HOST_DEVICE constexpr auto make_mesh(Extents const &extents) {
    static_assert(Scope == TopologyScope::cta || Scope == TopologyScope::thread,
                  "make_mesh: scope must be cta or thread");
    auto layout = cute::make_layout(extents, cute::GenRowMajor{});
    using level_t = cute::remove_cvref_t<decltype(program_dim<Scope>())>;
    if constexpr (cute::is_static<Extents>::value &&
                  cute::is_static<level_t>::value)
        static_assert(int(decltype(cute::size(layout))::value) <=
                          int(level_t{}),
                      "make_mesh: a mesh cannot exceed its topology level");
    return Mesh<decltype(layout), Scope>{layout};
}
