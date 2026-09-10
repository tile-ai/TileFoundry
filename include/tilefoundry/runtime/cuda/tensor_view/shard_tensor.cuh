/// CUDA ShardTensor tensor-view helpers. Included in-context from runtime.cuh
/// inside namespace tilefoundry.
#pragma once

/// A tensor, the global layout it came from, and how a mesh spreads it.
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
    typename SL::mesh::layout_type{}))>;

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
    static_assert(shard_mesh_axes_over<SL, Axis>(seq{}) <= 1,
                  "two mesh axes name one tensor axis");
    return shard_mesh_axis_over<SL, Axis>(seq{});
}

/// Local extent of tensor axis I.
template <size_t I, class L, class A, class M>
CUTE_HOST_DEVICE constexpr auto local_extent(ShardLayout<L, A, M> const &sl) {
    auto const ext = cute::get<I>(cute::shape(sl.layout_value));
    constexpr int m = shard_mesh_axis<ShardLayout<L, A, M>, int(I)>();
    if constexpr (m < 0)
        return ext;
    else
        return ext / cute::get<size_t(m)>(cute::shape(sl.mesh_value.layout));
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
        return local_extent<size_t(k)>(sl) * local_stride<size_t(k)>(sl);
    } else {
        static_assert(attr_leaves_tensor_whole<attr_t>(),
                      "shard_offset: this attr must leave the tensor whole");
        return cute::Int<0>{};
    }
}

/// A layout over the mesh whose values are storage offsets: the shard layout
/// read as the map from an instance to where that instance's slice starts.
template <class L, class A, class M>
CUTE_HOST_DEVICE constexpr auto offset_layout(ShardLayout<L, A, M> const &sl) {
    static_assert(
        shard_attrs_match_mesh<ShardLayout<L, A, M>>(),
        "one attr per mesh axis: this layout has one stride per attr");
    return [&]<size_t... Ax>(std::index_sequence<Ax...>) {
        return cute::make_layout(
            cute::shape(sl.mesh_value.layout),
            cute::make_stride(mesh_axis_stride<Ax>(sl)...));
    }(std::make_index_sequence<cute::tuple_size<A>::value>{});
}

/// This instance's element offset into a ShardTensor's engine.
template <class L, class A, class M>
CUTE_HOST_DEVICE int shard_offset(ShardLayout<L, A, M> const &sl) {
    return int(offset_layout(sl)(
        tilefoundry::get_1d_coord(sl.mesh_value, tilefoundry::program_ids())));
}

template <class T> struct is_shard_tensor : std::false_type {};
template <class E, class GL, class SL>
struct is_shard_tensor<ShardTensor<E, GL, SL>> : std::true_type {};

/// The one test for "is this operand sharded".
template <class T>
concept ShardTensorLike = is_shard_tensor<cute::remove_cvref_t<T>>::value;

/// Project one mesh instance's slice. A shard layout no mesh axis splits
/// leaves every instance the whole tensor, so there is nothing to project.
template <class T, class GL, class SL>
CUTE_HOST_DEVICE auto local(ShardTensor<T, GL, SL> const &st) {
    if constexpr (shard_layout_is_full_broadcast<SL>()) {
        return st.engine;
    } else {
        constexpr int t_rank =
            cute::tuple_size<cute::remove_cvref_t<decltype(cute::shape(
                typename SL::layout{}))>>::value;
        const int off = shard_offset(st.shard_layout);
        auto loc_layout = [&]<size_t... Is>(std::index_sequence<Is...>) {
            return cute::make_layout(
                cute::make_shape(local_extent<Is>(st.shard_layout)...),
                cute::make_stride(local_stride<Is>(st.shard_layout)...));
        }(std::make_index_sequence<t_rank>{});
        auto &engine_mut = const_cast<typename std::remove_const<
            typename std::remove_reference<decltype(st.engine)>::type>::type &>(
            st.engine);
        return cute::make_tensor(engine_mut.data() + off, loc_layout);
    }
}

/// A ShardTensor resolved to this instance's slice; anything else unchanged.
template <class T> CUTE_HOST_DEVICE decltype(auto) to_local(T &&t) {
    if constexpr (is_shard_tensor<cute::remove_cvref_t<T>>::value)
        return local(t);
    else
        return std::forward<T>(t);
}

/// The same answer as a type.
template <class T>
using local_view_t =
    cute::remove_cvref_t<decltype(to_local(std::declval<T const &>()))>;

}

/// How many instances the mesh of ``T``'s shard layout spreads it over.
template <class T> CUTE_HOST_DEVICE constexpr int shard_mesh_instances() {
    using mesh_t = typename cute::remove_cvref_t<T>::shard_layout_type::mesh;
    detail::one_level<mesh_t>();
    return int(cute::size(typename mesh_t::layout_type{}));
}
