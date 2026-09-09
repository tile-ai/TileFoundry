/// CUDA ShardTensor tensor-view helpers. Included in-context from runtime.cuh
/// inside namespace tilefoundry.
#pragma once

template <class TEngine, class TGlobalLayout, class TShardLayout>
struct ShardTensor {
    using engine_type = TEngine;
    using global_layout_type = TGlobalLayout;
    using shard_layout_type = TShardLayout;
    TEngine engine;
    TShardLayout shard_layout;

    CUTE_HOST_DEVICE auto data() { return engine.data(); }
    CUTE_HOST_DEVICE auto data() const { return engine.data(); }
};

template <class T, class GL, class SL>
CUTE_HOST_DEVICE auto make_shard_tensor(T const &tensor, GL, SL shard_layout) {
    using engine_t = cute::remove_cvref_t<T>;
    static_assert(
        !std::is_pointer_v<engine_t>,
        "ShardTensor engine must be a CuTe tensor/view, not a raw pointer");
    return ShardTensor<T, GL, SL>{tensor, shard_layout};
}

namespace detail {

/// The mesh shape a ShardLayout's attrs are indexed against.
template <class SL>
using shard_mesh_shape_t = cute::remove_cvref_t<decltype(cute::shape(
    detail::mesh_positions_t<typename SL::mesh::layout>{}))>;

/// The mesh axis that splits tensor axis ``Axis``, or -1 when none does.
///
/// ``attrs`` is indexed by mesh axis and each ``S<k>`` names a tensor axis, so
/// the map read this way round is a search. It runs over types, in constant
/// expressions: the alternative -- an ``int`` table filled from those same
/// types and then indexed in a loop -- cannot be a template argument, and every
/// extent and stride reached through it had to become an ``int`` as well.
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

/// How many mesh axes name tensor axis ``Axis``. One is the only answer the
/// sum below is written for.
template <class SL, int Axis, size_t... Is>
CUTE_HOST_DEVICE constexpr int
shard_mesh_axes_over(std::index_sequence<Is...>) {
    using attrs_t = typename SL::attrs;
    return (0 + ... +
            int(attr_axis<cute::remove_cvref_t<decltype(cute::get<Is>(
                    attrs_t{}))>>::value == Axis));
}

/// The mesh axis that names tensor axis ``Axis``, and ``-1`` where none does.
///
/// One mesh axis per tensor axis is the only spelling read here: a two-level
/// split is two modes and one ``S`` per mode ([shard
/// §7.1.1](docs/spec/shard.md#711-layoutshape)), which is what every layout
/// here does, so nothing needs the other one.
template <class SL, int Axis> CUTE_HOST_DEVICE constexpr int shard_mesh_axis() {
    using seq = std::make_index_sequence<
        cute::tuple_size<shard_mesh_shape_t<SL>>::value>;
    static_assert(shard_mesh_axes_over<SL, Axis>(seq{}) <= 1,
                  "two mesh axes name one tensor axis; split it into one mode "
                  "per mesh axis and put one S<> on each -- the search here "
                  "answers with the last of the two, and then local_extent "
                  "divides that axis by one mesh extent while shard_offset "
                  "adds a term for both, so a (warps, lanes) mesh written "
                  "S<0>, S<0> strides each of its 256 threads by the box four "
                  "of them own");
    return shard_mesh_axis_over<SL, Axis>(seq{});
}

/// Tensor axis ``I``'s local extent: the shard layout's own, over however many
/// ways the mesh splits it.
///
/// Static wherever the types are. Both halves can be known -- the layout states
/// the extent and the mesh states the split -- and then the quotient is a
/// ``cute::Int`` too, which is what lets a vector width or a copy tier be read
/// off the local layout's type. A dynamic extent on either side sinks this one
/// mode to ``int``; its neighbours keep what they knew, and the divisor is
/// still a constant.
template <size_t I, class L, class A, class M>
CUTE_HOST_DEVICE constexpr auto local_extent(ShardLayout<L, A, M> const &sl) {
    auto const ext = cute::get<I>(cute::shape(sl.layout_value));
    constexpr int m = shard_mesh_axis<ShardLayout<L, A, M>, int(I)>();
    if constexpr (m < 0) {
        return ext;
    } else {
        auto const positions =
            detail::mesh_positions(sl.mesh_value.layout_value);
        return ext / cute::get<size_t(m)>(cute::shape(positions));
    }
}

/// Tensor axis ``I``'s local stride: the shard layout's own, splitting or not.
///
/// A shard hands out a strided window of the same *storage*, so the step
/// between neighbours never changes -- only where the window starts. Which
/// storage the layout has already said ([shard
/// §7.1.2](docs/spec/shard.md#712-layoutstrides)): a shared buffer carries the
/// buffer's steps and a per-instance engine its own. Nothing is edited here
/// either way, compile-time fact or not.
template <size_t I, class L, class A, class M>
CUTE_HOST_DEVICE constexpr auto local_stride(ShardLayout<L, A, M> const &sl) {
    return cute::get<I>(cute::stride(sl.layout_value));
}

/// This instance's coordinate in a ShardLayout's mesh.
///
/// ``program_id`` is an instance of the *launch*; a coordinate is an instance
/// of the *mesh*. On a mesh the whole level wide the two are the same number,
/// which is what makes the subtraction easy to leave out and impossible to
/// notice: a mesh narrowed to ``m[128:]`` starts at ``mesh_offset``, and
/// reading its coordinate off the raw thread id hands every instance the box
/// its neighbour owns.
template <class L, class A, class M>
CUTE_HOST_DEVICE auto shard_mesh_coord(ShardLayout<L, A, M> const &sl) {
    return sl.mesh_value.local_index();
}

/// Mesh axis ``Ax``'s term of the offset sum, and a static zero where it has
/// none.
///
/// The extent and the stride are parenthesised so their product folds first: a
/// Split axis is then one multiply of a coordinate by one constant. Whether an
/// axis has a term is the attr's own kind, asked through
/// ``attr_leaves_tensor_whole`` and not through whether it carries an ``axis``
/// member -- ``shard::Dynamic`` carries none either, so it used to fall in
/// with ``Broadcast`` and ``Partial`` and have its term dropped.
template <size_t Ax, class L, class A, class M, class Crd>
CUTE_HOST_DEVICE constexpr auto mesh_axis_term(ShardLayout<L, A, M> const &sl,
                                               Crd const &crd) {
    using attr_t = cute::remove_cvref_t<decltype(cute::get<Ax>(A{}))>;
    constexpr int k = attr_axis<attr_t>::value;
    if constexpr (k >= 0) {
        return cute::get<Ax>(crd) *
               (local_extent<size_t(k)>(sl) * local_stride<size_t(k)>(sl));
    } else {
        static_assert(attr_leaves_tensor_whole<attr_t>(),
                      "shard_offset: an attr that names a tensor axis is the "
                      "branch above, so anything reaching here has to be one "
                      "that leaves the tensor whole -- Broadcast and Partial "
                      "name none, which is what makes their term the Int<0> "
                      "the sum drops before any code is emitted");
        return cute::Int<0>{};
    }
}

/// This instance's element offset into a ShardTensor's engine.
///
/// ``off = sum of crd[a] * local_extent[k] * stride[k]`` over the mesh axes
/// whose attr is ``S<k>`` --
/// [runtime §2.10.2](docs/spec/runtime.md#2102-computation)'s sum with the
/// local extent written out, which
/// [shard §7.1.1](docs/spec/shard.md#711-layoutshape) pins to one.
///
/// Unrolled, so every factor but ``crd`` is folded before any code is emitted.
/// ``crd`` cannot be: it is this instance's identity, which no type states.
template <class L, class A, class M>
CUTE_HOST_DEVICE int shard_offset(ShardLayout<L, A, M> const &sl) {
    using SL = ShardLayout<L, A, M>;
    static_assert(
        shard_attrs_match_mesh<SL>(),
        "one attr per mesh axis: the sum below has one term per attr, "
        "so a rank the attrs and the mesh do not share is an offset "
        "missing a term or reaching for an axis the mesh has not got");
    auto const crd = shard_mesh_coord(sl);
    return int([&]<size_t... Ax>(std::index_sequence<Ax...>) {
        return (cute::Int<0>{} + ... + mesh_axis_term<Ax>(sl, crd));
    }(std::make_index_sequence<cute::tuple_size<A>::value>{}));
}

}

template <class T, class GL, class SL>
CUTE_HOST_DEVICE auto local_impl(ShardTensor<T, GL, SL> const &st,
                                 std::true_type) {
    return st.engine;
}

/// One mesh instance's slice: the shard layout divided by the mesh, offset to
/// this instance.
///
/// The layout is per-mode mixed on purpose. An extent or a stride stays a
/// ``cute::Int`` when the shard layout and the mesh already settled it, so a
/// caller downstream can still read a vector width off the type. The offset
/// gets no such treatment: it is a function of the mesh *coordinate*, which
/// only exists once a program is running, and it is an ``int`` -- so one
/// tensor is bounded at 2^31 elements, 4 GiB of bf16.
template <class T, class GL, class SL>
CUTE_HOST_DEVICE auto local_impl(ShardTensor<T, GL, SL> const &st,
                                 std::false_type) {
    using sl_shape_t =
        cute::remove_cvref_t<decltype(cute::shape(typename SL::layout{}))>;
    constexpr int t_rank = cute::tuple_size<sl_shape_t>::value;

    const int off = detail::shard_offset(st.shard_layout);
    auto loc_layout = [&]<size_t... Is>(std::index_sequence<Is...>) {
        return cute::make_layout(
            cute::make_shape(detail::local_extent<Is>(st.shard_layout)...),
            cute::make_stride(detail::local_stride<Is>(st.shard_layout)...));
    }(std::make_index_sequence<t_rank>{});

    auto &engine_mut = const_cast<typename std::remove_const<
        typename std::remove_reference<decltype(st.engine)>::type>::type &>(
        st.engine);
    return cute::make_tensor(engine_mut.data() + off, loc_layout);
}

/// One projection, whatever the storage is.
///
/// No gmem / smem / rmem test, and that is the point: ``S[k]`` steps the
/// storage the engine holds ([shard
/// §7.1.2](docs/spec/shard.md#712-layoutstrides)), so a register shard states
/// ``0`` on every Split axis and one formula serves all three ([runtime
/// §2.10.3](docs/spec/runtime.md#2103-single-path-across-storages)). What stood
/// here returned register engines unprojected: a layout carrying the *tile's*
/// strides asked for offset ``1014`` into an eight-float accumulator.
template <class T, class GL, class SL>
CUTE_HOST_DEVICE auto local(ShardTensor<T, GL, SL> const &st) {
    constexpr bool full_bc = detail::shard_layout_is_full_broadcast<SL>();
    return local_impl(st, std::bool_constant<full_bc>{});
}

namespace detail {

template <class T> struct is_shard_tensor : std::false_type {};
template <class E, class GL, class SL>
struct is_shard_tensor<ShardTensor<E, GL, SL>> : std::true_type {};

/// The one test for "is this operand sharded".
///
/// Matching the type, not probing for a ``shard_layout_type`` member: the two
/// differ on anything that merely carries such a member, and the ops that
/// probed each said yes where this says no. Ops constrain on this, so a new one
/// cannot introduce a fourth spelling by accident.
template <class T>
concept ShardTensorLike = is_shard_tensor<cute::remove_cvref_t<T>>::value;

/// A ShardTensor resolved to this instance's slice; anything else unchanged.
///
/// Beside ``local()`` because it is the same question, and asking it in two
/// places is how the type-level answer and the value-level one drift apart.
template <class T> CUTE_HOST_DEVICE decltype(auto) to_local(T &&t) {
    if constexpr (is_shard_tensor<cute::remove_cvref_t<T>>::value) {
        return local(t);
    } else {
        return std::forward<T>(t);
    }
}

/// The same answer as a type.
template <class T>
using local_view_t =
    cute::remove_cvref_t<decltype(to_local(std::declval<T const &>()))>;

}

/// How many instances the mesh of ``T``'s shard layout spreads it over.
///
/// The operand-shaped spelling of ``mesh_instances``, so an op that has a
/// ShardTensor in hand does not dig out the mesh type to ask. This is where a
/// tier's participant count comes from: ``ops::mma`` reads its warp count off
/// the accumulator and ``ops::dot`` off its left operand, both through here.
template <class T> CUTE_HOST_DEVICE constexpr int shard_mesh_instances() {
    using mesh_t = typename cute::remove_cvref_t<T>::shard_layout_type::mesh;
    return detail::mesh_instances<mesh_t, TopologyScope::thread>();
}

/// An index offset inside an already-projected destination view.
/// Partial-broadcast async copies use it to place each thread's source range.
///
/// The sum ``local()`` applies to the pointer, asked of the same
/// ``detail::shard_offset``. One place, because the local extent is exactly the
/// factor a second copy is free to omit:
/// [shard §7.1.1](docs/spec/shard.md#711-layoutshape) makes it one, so an
/// omission agrees with this everywhere the spec holds and nowhere else.
template <class T, class GL, class SL>
CUTE_HOST_DEVICE int local_offset(ShardTensor<T, GL, SL> const &st) {
    if constexpr (detail::shard_layout_is_full_broadcast<SL>()) {
        return 0;
    } else {
        return detail::shard_offset(st.shard_layout);
    }
}
