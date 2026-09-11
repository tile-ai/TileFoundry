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

    /// As ``cute::Tensor`` has it: the engine's own pointer, before any
    /// instance's slice is projected out of it.
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

template <class T> struct is_shard_tensor : std::false_type {};
template <class E, class GL, class SL>
struct is_shard_tensor<ShardTensor<E, GL, SL>> : std::true_type {};

}

/// The one test for "is this operand sharded". Public, because it is the
/// word an op writes its own constraints in; ``detail::is_shard_tensor`` is
/// how the test is made and stays behind the gate.
template <class T>
concept ShardTensorLike =
    detail::is_shard_tensor<cute::remove_cvref_t<T>>::value;

/// ``t`` as the tensor this instance holds, in CuTe's ``local_tile`` /
/// ``local_partition`` sense: a ShardTensor projected to its own slice, and
/// anything the mesh never spread already whole. An op takes both -- an
/// operand with no shard layout is not a case to reject, it is one every
/// instance holds entire -- and its arithmetic is the same either way.
///
/// Public, because every op projects every operand through it: a step none
/// may skip is no implementation detail. A layout no mesh axis splits leaves
/// every instance the whole tensor, so this is the only place an id is read.
template <class T> CUTE_HOST_DEVICE decltype(auto) local_tensor(T &&t) {
    using t_t = cute::remove_cvref_t<T>;
    if constexpr (!detail::is_shard_tensor<t_t>::value) {
        return std::forward<T>(t);
    } else if constexpr (detail::shard_layout_is_full_broadcast<
                             typename t_t::shard_layout_type>()) {
        return t.engine;
    } else {
        auto const [loc_layout, off] = detail::local_layout_and_offset(
            t.shard_layout,
            tilefoundry::mesh_coords(t.shard_layout.mesh_value,
                                     tilefoundry::program_ids()));
        auto &engine_mut = const_cast<typename std::remove_const<
            typename std::remove_reference<decltype(t.engine)>::type>::type &>(
            t.engine);
        return cute::make_tensor(engine_mut.data() + off, loc_layout);
    }
}

/// The same answer as a type, for a constraint written before there is a
/// value to project. Public with ``local_tensor``, of which it is the
/// type-level half.
template <class T>
using local_view_t =
    cute::remove_cvref_t<decltype(local_tensor(std::declval<T const &>()))>;

/// How many instances the mesh of ``T``'s shard layout spreads it over.
template <class T> CUTE_HOST_DEVICE constexpr int shard_mesh_instances() {
    using mesh_t = typename cute::remove_cvref_t<T>::shard_layout_type::mesh;
    return int(cute::size(typename mesh_t::layout_type{}));
}
