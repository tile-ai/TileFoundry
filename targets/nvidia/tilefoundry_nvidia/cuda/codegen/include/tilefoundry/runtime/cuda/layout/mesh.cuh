/// A mesh: the hardware topologies it names, and one CuTe layout whose value at
/// a coord is the unit's index over all those topologies, ``T`` the topologies'
/// sizes row-major. A topology's unit id at an index is ``idx2crd`` of it over
/// ``T``. The C++ half of [shard §5](docs/spec/shard.md#5-mesh).
///
/// The struct carries the two fields and nothing else. Everything asked of a
/// mesh is either a CuTe question about that layout -- ``cute::size`` /
/// ``shape`` / ``rank`` / evaluation, which read through a ``ComposedLayout``
/// slice on their own -- or one of the free functions below.
#pragma once

namespace detail {
/// The pack's own first value. Reading it off ``decltype(tuple){}`` instead
/// hands back the zeroth enumerator, which is a different topology.
template <TopologyLevel T0, TopologyLevel...> struct first_topology_of {
    static constexpr TopologyLevel value = T0;
};
}

/// The topologies, and the layout whose values are indices over all of them.
/// How many units a topology has is the launch's, not the mesh's: one .cu is
/// one launch, so
/// ``topology_size<topology>()`` states it and the same mesh in another
/// translation unit is another launch's mesh.
template <class TLayout, TopologyLevel... Topos> struct Mesh {
    using layout_type = TLayout;
    static constexpr int topology_count = sizeof...(Topos);
    static constexpr TopologyLevel first_topology =
        detail::first_topology_of<Topos...>::value;
    TLayout layout;

    /// The topology flat mesh axis ``I`` runs at. A mesh naming one topology
    /// answers its own topology for every mesh axis. A mesh naming several
    /// groups its layout one nest per topology, and the nest holding
    /// flat mesh axis ``I`` states its topology.
    template <int I>
    CUTE_HOST_DEVICE static constexpr TopologyLevel mesh_axis_topology() {
        if constexpr (sizeof...(Topos) == 1) {
            return first_topology;
        } else {
            constexpr TopologyLevel topologies[] = {Topos...};
            TopologyLevel found = first_topology;
            int first = 0;
            [&]<size_t... Ls>(std::index_sequence<Ls...>) {
                (([&] {
                     constexpr int width =
                         int(cute::rank(cute::get<Ls>(cute::shape(TLayout{}))));
                     if (I >= first && I < first + width)
                         found = topologies[Ls];
                     first += width;
                 }()),
                 ...);
            }(std::make_index_sequence<sizeof...(Topos)>{});
            return found;
        }
    }
};

namespace detail {

/// Where ``S`` sits in the pack, and ``sizeof...`` when it names no topology.
template <TopologyLevel S, TopologyLevel... Ts>
CUTE_HOST_DEVICE constexpr int topology_level_of() {
    constexpr bool hit[] = {(Ts == S)...};
    for (int i = 0; i < int(sizeof...(Ts)); ++i)
        if (hit[i])
            return i;
    return int(sizeof...(Ts));
}

/// The plain layout under a slice: what carries ``get_hier_coord``.
template <class L> CUTE_HOST_DEVICE constexpr auto plain_layout(L const &l) {
    if constexpr (cute::is_composed_layout<cute::remove_cvref_t<L>>::value)
        return l.layout_b();
    else
        return l;
}

/// The mesh's id axes, in the order CuTe's algebra reads them.
///
/// ``reverse`` because a mesh is row-major and CuTe is not -- the reason is
/// on ``reverse`` itself in layout/cute_ext.cuh -- and ``filter`` because a
/// mesh axis of extent one, which CuTe writes as stride zero, names no id.
template <class L> CUTE_HOST_DEVICE constexpr auto id_axes(L const &l) {
    return cute::filter(reverse(cute::flatten(plain_layout(l))));
}

/// The mode of the mesh's layout that holds topology ``S``'s mesh axes:
/// ``cute::get<at>`` of the plain layout, strides in the index over all the
/// mesh's topologies. A mesh naming one topology states its mesh axes
/// ungrouped, so its whole plain layout is that mode.
template <TopologyLevel S, class L, TopologyLevel... Topos>
CUTE_HOST_DEVICE constexpr auto topology_mode(Mesh<L, Topos...> const &mesh) {
    constexpr int at = topology_level_of<S, Topos...>();
    static_assert(at < int(sizeof...(Topos)),
                  "topology_mode: this mesh does not name that topology");
    if constexpr (sizeof...(Topos) == 1)
        return plain_layout(mesh.layout);
    else if constexpr (at < int(sizeof...(Topos)))
        return cute::get<at>(plain_layout(mesh.layout));
    else
        return plain_layout(mesh.layout);
}

/// The shape of ``T``, the index over the mesh's topologies. The outermost
/// size is never read: no stride holds it, and ``idx2crd`` keeps the
/// outermost quotient. A run-time 1 stands for it, so a translation unit
/// need not state it.
template <TopologyLevel T0, TopologyLevel... Inner>
CUTE_HOST_DEVICE constexpr auto index_shape() {
    return cute::make_shape(1, topology_size<Inner>()...);
}

/// ``stride(T)``: each topology's stride in the index, row-major.
template <TopologyLevel... Topos>
CUTE_HOST_DEVICE constexpr auto topology_strides() {
    return cute::compact_row_major(index_shape<Topos...>());
}

/// Each topology's unit id at the mesh's first unit: ``idx2crd(layout(0), T)``
/// in its two-argument form, row-major, so the outermost topology keeps the
/// quotient. A plain layout starts at index 0, every id 0.
template <class L, TopologyLevel... Topos>
CUTE_HOST_DEVICE constexpr auto first_ids(Mesh<L, Topos...> const &mesh) {
    if constexpr (!cute::is_composed_layout<cute::remove_cvref_t<L>>::value)
        return cute::repeat<sizeof...(Topos)>(0);
    else
        return cute::reverse(
            cute::idx2crd(mesh.layout(cute::Int<0>{}),
                          cute::reverse(index_shape<Topos...>())));
}

/// Topology ``S``'s unit id at the mesh's first unit.
template <TopologyLevel S, class L, TopologyLevel... Topos>
CUTE_HOST_DEVICE constexpr int first_id(Mesh<L, Topos...> const &mesh) {
    return int(cute::get<topology_level_of<S, Topos...>()>(first_ids(mesh)));
}

/// A coord states one id per topology this program names, coarsest first,
/// which is what ``unit_ids()`` hands back. A shorter one is missing a
/// topology rather than naming a smaller mesh, and reads a neighbour's id if
/// left to index.
template <class Coord> CUTE_HOST_DEVICE constexpr void one_id_per_topology() {
    static_assert(cute::tuple_size<cute::remove_cvref_t<Coord>>::value ==
                      size_t(cute::rank(program_topologies())),
                  "coord: one id per topology named, as unit_ids() gives");
}

/// Topology ``S``'s id in ``coord``, counted from the mesh's first unit and
/// stepped by ``S``'s stride in ``T``: an index ``S``'s mode can read back.
template <TopologyLevel S, class L, TopologyLevel... Topos, class Coord>
CUTE_HOST_DEVICE constexpr int id_index(Mesh<L, Topos...> const &mesh,
                                        Coord const &coord) {
    constexpr int t = topology_level_of<S, Topos...>();
    constexpr auto at = cute::find(program_topologies(), cute::C<S>{});
    return (int(cute::get<at>(coord)) - first_id<S>(mesh)) *
           int(cute::get<t>(topology_strides<Topos...>()));
}

/// Whether topology ``S``'s mode reaches the id ``coord`` gives ``S``.
template <TopologyLevel S, class L, TopologyLevel... Topos, class Coord>
CUTE_HOST_DEVICE constexpr bool covers(Mesh<L, Topos...> const &mesh,
                                       Coord const &coord) {
    const int idx = id_index<S>(mesh, coord);
    if (idx < 0)
        return false;
    auto const mode = topology_mode<S>(mesh);
    auto const at = mode.get_hier_coord(idx);
    /// idx2crd answers for a hole and for overshoot too; only the value
    /// it was built to reproduce says the id is one this mesh covers.
    return cute::elem_less(at, cute::shape(mode)) && int(mode(at)) == idx;
}
}

/// Whether ``coord`` names a unit of ``mesh``.
///
/// The coord is one id per topology named, which is what ``unit_ids()``
/// hands back. Each topology the mesh names is read alone, in its own mode
/// with its strides in the index over all the mesh's topologies, and has to
/// cover its own id.
template <class L, TopologyLevel... Topos, class Coord>
CUTE_HOST_DEVICE constexpr bool contains(Mesh<L, Topos...> const &mesh,
                                         Coord const &coord) {
    detail::one_id_per_topology<Coord>();
    return (detail::covers<Topos>(mesh, coord) && ...);
}

/// Which unit ``coord`` acts as, in the mesh's own coordinates.
///
/// One ``idx2crd`` per topology through that topology's mode, grouped the way
/// the mesh is, so the answer is congruent with ``shape(mesh.layout)``.
///
/// Not ``contains``: ``idx2crd`` is ``(id / stride) % extent``, so thread 64
/// of a 128-thread block acts as unit 0 of a 32-unit mesh it is not
/// in. A slice does not repeat -- its body runs under ``contains`` -- so a
/// coord from outside one reaching here is a codegen fault.
template <class L, TopologyLevel... Topos, class Coord>
CUTE_HOST_DEVICE constexpr auto mesh_coords(Mesh<L, Topos...> const &mesh,
                                            Coord const &coord) {
    detail::one_id_per_topology<Coord>();
    if constexpr (cute::is_composed_layout<cute::remove_cvref_t<L>>::value)
        assert(contains(mesh, coord));
    if constexpr (sizeof...(Topos) > 1)
        return cute::make_tuple(
            detail::topology_mode<Topos>(mesh).get_hier_coord(
                detail::id_index<Topos>(mesh, coord))...);
    else
        return detail::plain_layout(mesh.layout)
            .get_hier_coord(detail::id_index<Mesh<L, Topos...>::first_topology>(
                mesh, coord));
}

/// Whether the mesh's ids say where the warps are.
///
/// The mesh axes arrive in CuTe's order, so mode zero is the fastest and its
/// extent is how many lanes run together: ``(32,..):(1,..)`` is one warp and
/// the next mesh axis carries the warps, ``(128,..):(1,..)`` holds four that
/// ``as_warped`` splits out, and ``(16,..):(1,..)`` is half a warp shared
/// between units. The third reaches neither ``bar.sync``, which counts
/// whole warps, nor one ``__syncwarp``, so it is refused by name.
template <class L, TopologyLevel... Topos>
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

/// The same mesh with its lanes stated as a mesh axis of their own: the fastest
/// mesh axis of extent ``E`` becomes ``(E/32, 32)``, so the last mesh axis is
/// one warp and the mesh axis before it steps between warps.
///
/// Idempotent -- a fastest mesh axis of exactly 32 is already that -- and every
/// warp question is then a shape or a stride of the result.
template <class L, TopologyLevel... Topos>
CUTE_HOST_DEVICE constexpr auto as_warped(Mesh<L, Topos...> const &mesh) {
    static_assert(is_warped(Mesh<L, Topos...>{}),
                  "as_warped: this mesh's fastest axis must step by one "
                  "and run whole warps");
    using ids_t = cute::remove_cvref_t<decltype(detail::id_axes(mesh.layout))>;
    /// Mode zero is the fastest here, so a warp-sized tile cuts the mesh axis a
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
template <TopologyLevel S, class Extents>
CUTE_HOST_DEVICE constexpr auto make_mesh(Extents const &extents) {
    static_assert(S == TopologyLevel::cta || S == TopologyLevel::thread,
                  "make_mesh: topology must be cta or thread");
    auto layout = cute::make_layout(extents, cute::GenRowMajor{});
    using size_type = cute::remove_cvref_t<decltype(topology_size<S>())>;
    if constexpr (cute::is_static<Extents>::value &&
                  cute::is_static<size_type>::value)
        static_assert(int(decltype(cute::size(layout))::value) <=
                          int(size_type{}),
                      "make_mesh: a mesh cannot exceed its topology");
    return Mesh<decltype(layout), S>{layout};
}
