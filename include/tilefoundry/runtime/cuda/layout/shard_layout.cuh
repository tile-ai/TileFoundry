/// CUDA shard layout surface: the types, and the constructors that build them.
/// Included in-context from runtime.cuh inside namespace tilefoundry.
#pragma once

/// Which level of the launch a mesh spreads over. Nothing else.
///
/// How many instances there are, and how they are shaped, is the mesh layout's
/// answer -- ``cute::size(mesh.layout)``. Carrying a second copy here made the
/// two disagreeable: ``Mesh`` is an aggregate, so a hand-written one could
/// state 256 in its topology and a 64-element layout, and nothing checked.
template <TopologyScope Scope> struct Topology {
    static constexpr TopologyScope scope = Scope;
};

/// Mesh<topology, cute_layout>: a topology bound to a MeshLayout -- the two
/// fields the IR ``Mesh`` has beside its axis names.
///
/// A narrowed scope -- ``m[128:]`` on a 256-thread block -- is not a third
/// field. The IR spells it ``ComposedLayout(inner, offset, outer)``; this
/// mirrors it with the CuTe type of that name, whose ``operator()`` is
/// ``layout_a()(offset() + layout_b()(c))``. A ``Base`` argument would say
/// that in a second place, free to disagree, where the IR has no field at
/// all; a plain ``cute::Layout`` is the whole level at offset zero.
template <class TMeshLayout, TopologyScope... Topos> struct Mesh {
    using layout = TMeshLayout;
    static constexpr auto topologies = cute::make_tuple(Topos...);
    static constexpr TopologyScope first_scope = cute::get<0>(topologies);
    TMeshLayout layout_value;
    CUTE_HOST_DEVICE auto local_index() const noexcept;
};

namespace detail {

/// The constant term of a mesh layout: zero unless it is a composed one.
///
/// Only the shape CuTe's ``ComposedLayout`` takes here is admitted -- identity
/// over a static offset -- because that is the one the IR's ``ComposedLayout``
/// lowers to. A swizzle in the first slot, or a dynamic offset, is a mesh whose
/// first instance is not a compile-time number, and every reader below wants it
/// as one.
template <class L> struct mesh_layout_offset {
    static constexpr int value = 0;
};
/// Every ``ComposedLayout`` matches, and the two admitted properties are
/// asserted rather than pattern-matched.
///
/// Matching only ``<identity, Int<n>, B>`` left every other composition on the
/// primary template above, whose answer is ``0``: a swizzled mesh, or one whose
/// offset is a run-time value, silently became a mesh based at instance zero
/// and handed every instance the box the first one owns.
template <class A, class O, class B>
struct mesh_layout_offset<cute::ComposedLayout<A, O, B>> {
    static_assert(std::is_same_v<cute::remove_cvref_t<A>, cute::identity>,
                  "mesh_offset: a mesh layout's first component must be "
                  "cute::identity -- a swizzle there is a mesh whose first "
                  "instance is not a number, and every reader wants it as one");
    static_assert(cute::is_static<cute::remove_cvref_t<O>>::value,
                  "mesh_offset: a mesh layout's offset must be a compile-time "
                  "number -- the IR's ComposedLayout lowers a slice origin to "
                  "cute::Int<n>, and a dynamic one cannot be a tier's or a "
                  "barrier's template argument");
    static constexpr int value = int(cute::remove_cvref_t<O>{});
};

}

namespace detail {
/// The first instance ``TMeshLayout`` covers, in the topology's own numbering.
template <class TMeshLayout> CUTE_HOST_DEVICE constexpr int mesh_offset() {
    return detail::mesh_layout_offset<cute::remove_cvref_t<TMeshLayout>>::value;
}

/// How many instances ``TMesh`` covers, as a compile-time number.
///
/// Beside ``mesh_offset`` because it is the other half of the same question --
/// where the mesh starts and how far it runs -- and because an op that needs it
/// must not go asking the launch: ``blockDim`` is the block and a mesh may be
/// narrower than one, so the two agree only where the mesh is the whole level.
template <class TMesh, TopologyScope Scope>
CUTE_HOST_DEVICE constexpr int mesh_instances();

template <class TMesh, TopologyScope Scope>
CUTE_HOST_DEVICE constexpr bool mesh_names_level() {
    return
        []<TopologyScope... Values>(Mesh<typename TMesh::layout, Values...> *) {
            return ((Values == Scope) || ...);
        }(static_cast<TMesh *>(nullptr));
}

template <class TMesh, TopologyScope... Scopes>
CUTE_HOST_DEVICE constexpr bool mesh_names_only() {
    return (mesh_names_level<TMesh, Scopes>() || ...);
}

template <class TMesh, TopologyScope Scope> struct mesh_axes_of_impl {
    using type = typename TMesh::layout;
};
template <class TMesh, TopologyScope Scope>
using mesh_axes_of = typename mesh_axes_of_impl<TMesh, Scope>::type;

template <class TMesh> CUTE_HOST_DEVICE constexpr void reject_multi_level() {
    static_assert(cute::tuple_size<decltype(TMesh::topologies)>::value == 1,
                  "Mesh: a mesh naming multiple topology levels is temporarily "
                  "unsupported; its axis assignment is not implemented");
}

template <class TMesh, TopologyScope Scope>
CUTE_HOST_DEVICE constexpr int mesh_instances() {
    reject_multi_level<TMesh>();
    return int(cute::size(detail::mesh_axes_of<TMesh, Scope>{}));
}

/// A mesh layout's positions, with its offset taken off.
///
/// What is left maps a mesh coordinate to an instance *within* the mesh, which
/// is what a coordinate query wants; the offset is what turns that into an
/// instance of the launch. Handing back the plain layout for an un-offset mesh
/// keeps one spelling at every call site.
template <class L>
CUTE_HOST_DEVICE constexpr auto mesh_positions(L const &layout) {
    if constexpr (cute::is_composed_layout<cute::remove_cvref_t<L>>::value)
        return layout.layout_b();
    else
        return layout;
}

/// The same answer as a type.
template <class L>
using mesh_positions_t =
    cute::remove_cvref_t<decltype(mesh_positions(std::declval<L const &>()))>;

}

template <class TMesh, TopologyScope... Topos>
CUTE_HOST_DEVICE auto Mesh<TMesh, Topos...>::local_index() const noexcept {
    detail::reject_multi_level<Mesh<TMesh, Topos...>>();
    using levels = decltype(Mesh<TMesh, Topos...>::topologies);
    constexpr auto scope = Mesh<TMesh, Topos...>::first_scope;
    auto const positions = detail::mesh_positions(this->layout_value);
    return positions.get_hier_coord(int(program_id<scope>()) -
                                    detail::mesh_offset<TMesh>());
}

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

/// The two attr kinds an offset has to tell apart from a Split, in one place:
/// every reader of ``attrs`` -- the offset sum, the reduce dispatch -- asks the
/// same question, and asking it twice is how one of them came to answer
/// ``shard::Dynamic`` as a broadcast while the other did not.
template <class A> struct is_split_attr : std::false_type {};
template <int Axis> struct is_split_attr<shard::S<Axis>> : std::true_type {};
template <class A>
inline constexpr bool is_split_attr_v = is_split_attr<A>::value;

template <class A> struct is_partial_attr : std::false_type {};
template <class R> struct is_partial_attr<shard::P<R>> : std::true_type {};
template <class A>
inline constexpr bool is_partial_attr_v = is_partial_attr<A>::value;

/// Whether a mesh axis's attr leaves every instance the whole of the tensor
/// axes, rather than dividing one of them.
///
/// Exhaustive on the attr's own kind, and the one place that is. Both readers
/// of ``attrs`` used to ask instead whether the attr *named* an axis, and
/// ``shard::Dynamic`` -- a split whose factor is not a compile-time fact --
/// names none: the full-broadcast test answered it as a broadcast and the
/// offset sum dropped its term, so a shard whose offset the layout could not
/// state read from offset zero on every instance.
template <class A> CUTE_HOST_DEVICE constexpr bool attr_leaves_tensor_whole() {
    if constexpr (is_split_attr_v<A>) {
        return false;
    } else if constexpr (std::is_same_v<A, shard::B> || is_partial_attr_v<A>) {
        /// ``B`` gives the instance everything; ``P`` gives it a contribution
        /// to all of the same elements. Neither moves where the slice starts.
        return true;
    } else {
        static_assert(dependent_false_v<A>,
                      "shard layout: this attr neither names a tensor axis "
                      "(shard::S<n>) nor leaves the tensor whole (shard::B, "
                      "shard::P) -- a shard::Dynamic split has no compile-time "
                      "offset, and reading it as a broadcast gives every "
                      "instance the first one's slice");
        return true;
    }
}

/// How many mesh axes a shard layout's attrs must speak for.
template <class SL> CUTE_HOST_DEVICE constexpr int shard_mesh_rank() {
    return int(cute::tuple_size<cute::remove_cvref_t<decltype(cute::shape(
                   mesh_positions_t<typename SL::mesh::layout>{}))>>::value);
}

/// Whether a shard layout says one thing about each of its mesh's axes.
template <class SL> CUTE_HOST_DEVICE constexpr bool shard_attrs_match_mesh() {
    return int(cute::tuple_size<typename SL::attrs>::value) ==
           shard_mesh_rank<SL>();
}

/// A shard layout no mesh axis splits: every instance holds the whole tensor.
///
/// Spelled by the attrs, one ``B`` (or ``P``) per mesh axis, and not by their
/// *number*. It used to read ``size == 0 || size != mesh rank``, so any
/// mismatch -- two attrs for a three-axis mesh -- was answered as a full
/// broadcast, and ``detail::shard_offset``'s own "one attr per mesh axis"
/// assertion became unreachable because callers routed the mismatch here
/// first. The mismatch is refused at construction and again here, for the
/// layouts codegen builds by naming the type directly.
template <class SL>
CUTE_HOST_DEVICE constexpr bool shard_layout_is_full_broadcast() {
    static_assert(
        shard_attrs_match_mesh<SL>(),
        "one attr per mesh axis: a shard layout must say what each "
        "of its mesh's axes does with the tensor, and a full "
        "broadcast is spelled by giving every axis shard::B -- not by "
        "leaving attrs short, which was read as one and quietly gave "
        "every instance the whole tensor");
    using attrs_t = typename SL::attrs;
    return [&]<size_t... Is>(std::index_sequence<Is...>) {
        return (
            true && ... &&
            attr_leaves_tensor_whole<
                cute::remove_cvref_t<decltype(cute::get<Is>(attrs_t{}))>>());
    }(std::make_index_sequence<cute::tuple_size<attrs_t>::value>{});
}

}

/// Both are aggregates, so building one meant writing its shape and its strides
/// out by hand -- the factorisation
/// [shard §7.1.1](docs/spec/shard.md#711-layoutshape) already specifies, redone
/// at every call site. These are the C++ twins of the Python ``make_mesh`` and
/// ``canonical_shard_layout``: a caller states the tensor's own layout, who
/// owns which axis, and how many elements an instance takes before it strides.
/// The factored shape is the answer, not the input.

/// A mesh over ``extents``, row-major, as the Python ``make_mesh`` builds one.
///
/// The stride order is what turns a linear instance id into a coordinate:
/// ``(8, 32)`` row-major gives ``(id / 32, id % 32)``, so a thread mesh names
/// its warps first and its lanes last.
///
/// A mesh wider than its level is refused here rather than read out of range
/// later: ``program_shape<Scope>()`` is what the launch states the level is, so
/// nothing else has to be told, and every ``shard_mesh_coord`` on an oversized
/// mesh indexes past the last instance the level has.
template <TopologyScope Scope, class Extents>
CUTE_HOST_DEVICE constexpr auto make_mesh(Extents const &extents) {
    static_assert(Scope == TopologyScope::cta || Scope == TopologyScope::thread,
                  "make_mesh: only the cta and thread levels have a shape to "
                  "spread over -- the warp level has no program_shape and "
                  "scope_count is a sentinel, so a warp-sized grouping belongs "
                  "as an axis of a thread mesh's layout");
    auto layout = cute::make_layout(extents, cute::GenRowMajor{});
    /// The same condition again as an ``if constexpr``, so that a scope with no
    /// shape is reported by the sentence above and not a second time by the
    /// failure to deduce one. ``Scope`` is this template's own parameter, which
    /// is what defers ``program_shape<Scope>`` to the call site, after the
    /// module has specialised it; a launch-sized level states no static shape
    /// and leaves nothing to compare.
    if constexpr (Scope == TopologyScope::cta ||
                  Scope == TopologyScope::thread) {
        using level_t = cute::remove_cvref_t<decltype(program_shape<Scope>())>;
        if constexpr (cute::is_static<Extents>::value &&
                      cute::is_static<level_t>::value) {
            static_assert(
                int(decltype(cute::size(layout))::value) <=
                    int(decltype(cute::size(level_t{}))::value),
                "make_mesh: a mesh cannot cover more instances than its "
                "topology level has -- program_shape<Scope>() states the "
                "level, and a coordinate on a wider mesh reads past its "
                "last instance");
        }
    }
    return Mesh<decltype(layout), Scope>{layout};
}

/// `make_shard_layout` is to `ShardLayout` what `cute::make_layout` is to
/// `Layout`: it packages the pieces and does not reinterpret the shape it is
/// handed. The axes are the caller's and the attrs point at them as written.
///
/// Two forms, because `cute::make_layout` already has the rest: a shape, whose
/// strides come out row-major, or a layout the caller built with whatever
/// strides or `GenRowMajor` / `GenColMajor` it wanted.

/// The one rule both forms impose, stated where the layout is built.
///
/// An attrs tuple shorter or longer than the mesh's rank is not a shorthand for
/// anything: ``detail::shard_offset`` sums one term per attr and
/// ``shard_mesh_axis`` searches all of them, so a rank the two do not share is
/// a layout whose offset is missing a term or reaching for an axis the mesh
/// does not have. Refusing it at construction is what makes both those
/// downstream assertions reachable claims rather than comments.
template <class SL> CUTE_HOST_DEVICE constexpr void check_shard_layout() {
    static_assert(detail::shard_attrs_match_mesh<SL>(),
                  "make_shard_layout: one attr per mesh axis -- a shorter or "
                  "longer attrs tuple is not a shorthand for a full broadcast; "
                  "spell that by giving every mesh axis shard::B");
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
