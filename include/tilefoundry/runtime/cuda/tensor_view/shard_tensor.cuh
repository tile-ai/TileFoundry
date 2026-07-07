// CUDA ShardTensor tensor-view helpers. Included in-context from runtime.cuh
// inside namespace tilefoundry.
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
CUTE_HOST_DEVICE auto make_shard_tensor(T const &tensor, GL /*global_layout*/,
                                        SL shard_layout) {
    using engine_t = cute::remove_cvref_t<T>;
    static_assert(
        !std::is_pointer_v<engine_t>,
        "ShardTensor engine must be a CuTe tensor/view, not a raw pointer");
    return ShardTensor<T, GL, SL>{tensor, shard_layout};
}

template <class T, class GL, class SL>
CUTE_HOST_DEVICE auto local_impl(ShardTensor<T, GL, SL> const &st,
                                 std::true_type /*full_broadcast*/) {
    return st.engine;
}

template <class T, class GL, class SL>
CUTE_HOST_DEVICE auto local_impl(ShardTensor<T, GL, SL> const &st,
                                 std::false_type /*full_broadcast*/) {
    using mesh_t = typename SL::mesh;
    using topo_t = typename mesh_t::topology;
    constexpr auto scope = topo_t::scope;
    using attrs_t = typename SL::attrs;
    using sl_layout_t = typename SL::layout;
    using m_layout_t = typename mesh_t::layout;

    auto const &sl_layout = st.shard_layout.layout_value;
    auto const &m_layout = st.shard_layout.mesh_value.layout_value;

    auto pid = program_id<scope>();
    auto crd = m_layout.get_hier_coord(pid);
    using sl_shape_t =
        cute::remove_cvref_t<decltype(cute::shape(sl_layout_t{}))>;
    using m_shape_t = cute::remove_cvref_t<decltype(cute::shape(m_layout_t{}))>;
    constexpr int t_rank = cute::tuple_size<sl_shape_t>::value;
    constexpr int m_rank = cute::tuple_size<m_shape_t>::value;

    auto const sl_shape = cute::shape(sl_layout);
    auto const sl_stride = cute::stride(sl_layout);
    auto const m_shape = cute::shape(m_layout);
    int g_dim[t_rank];
    int g_stride[t_rank];
    [&]<size_t... Is>(std::index_sequence<Is...>) {
        ((g_dim[Is] = int(cute::get<Is>(sl_shape)),
          g_stride[Is] = int(cute::get<Is>(sl_stride))),
         ...);
    }(std::make_index_sequence<t_rank>{});

    int m_ext[m_rank], m_crd[m_rank];
    [&]<size_t... Is>(std::index_sequence<Is...>) {
        ((m_ext[Is] = int(cute::get<Is>(m_shape)),
          m_crd[Is] = int(cute::get<Is>(crd))),
         ...);
    }(std::make_index_sequence<m_rank>{});

    int axis_to_mesh[t_rank];
    for (int i = 0; i < t_rank; ++i)
        axis_to_mesh[i] = -1;
    [&]<size_t... Is>(std::index_sequence<Is...>) {
        auto process = [&]<size_t I>(std::integral_constant<size_t, I>) {
            auto attr = cute::get<I>(attrs_t{});
            using A = decltype(attr);
            if constexpr (!std::is_same_v<A, shard::B>) {
                constexpr int ax = A::axis;
                axis_to_mesh[ax] = int(I);
            }
        };
        (process(std::integral_constant<size_t, Is>{}), ...);
    }(std::make_index_sequence<m_rank>{});

    int loc_shape[t_rank];
    int loc_stride[t_rank];
    int offset = 0;
    for (int i = 0; i < t_rank; ++i) {
        int m = axis_to_mesh[i];
        if (m >= 0) {
            offset += m_crd[m] * g_stride[i];
            loc_shape[i] = g_dim[i] / m_ext[m];
            loc_stride[i] = g_stride[i];
        } else {
            loc_shape[i] = g_dim[i];
            loc_stride[i] = g_stride[i];
        }
    }
    auto local_layout = [&]<size_t... Is>(std::index_sequence<Is...>) {
        return cute::make_layout(cute::make_shape(loc_shape[Is]...),
                                 cute::make_stride(loc_stride[Is]...));
    }(std::make_index_sequence<t_rank>{});

    auto &engine_mut = const_cast<typename std::remove_const<
        typename std::remove_reference<decltype(st.engine)>::type>::type &>(
        st.engine);
    return cute::make_tensor(engine_mut.data() + offset,
                             cute::coalesce(local_layout));
}

template <class T, class GL, class SL>
CUTE_HOST_DEVICE decltype(auto) local(ShardTensor<T, GL, SL> const &st) {
    using engine_t = cute::remove_cvref_t<decltype(st.engine)>;
    if constexpr (!cute::is_gmem<engine_t>::value &&
                  !cute::is_smem<engine_t>::value) {
        return const_cast<engine_t &>(st.engine);
    } else {
        using attrs_t = typename SL::attrs;
        using mesh_t = typename SL::mesh;
        using m_layout_t = typename mesh_t::layout;
        using m_shape_t =
            cute::remove_cvref_t<decltype(cute::shape(m_layout_t{}))>;
        constexpr bool full_bc = (cute::tuple_size<attrs_t>::value == 0) ||
                                 (cute::tuple_size<attrs_t>::value !=
                                  cute::tuple_size<m_shape_t>::value);
        return local_impl(st, std::bool_constant<full_bc>{});
    }
}

template <class T, class GL, class SL>
CUTE_HOST_DEVICE int local_offset(ShardTensor<T, GL, SL> const &st) {
    using mesh_t = typename SL::mesh;
    using attrs_t = typename SL::attrs;
    using sl_layout_t = typename SL::layout;
    using m_layout_t = typename mesh_t::layout;
    using topo_t = typename mesh_t::topology;
    constexpr auto scope = topo_t::scope;
    using sl_shape_t =
        cute::remove_cvref_t<decltype(cute::shape(sl_layout_t{}))>;
    using m_shape_t = cute::remove_cvref_t<decltype(cute::shape(m_layout_t{}))>;
    constexpr int t_rank = cute::tuple_size<sl_shape_t>::value;
    constexpr int m_rank = cute::tuple_size<m_shape_t>::value;
    if constexpr (cute::tuple_size<attrs_t>::value == 0 ||
                  cute::tuple_size<attrs_t>::value != m_rank) {
        return 0;
    } else {
        auto const &sl_layout = st.shard_layout.layout_value;
        auto const &m_layout = st.shard_layout.mesh_value.layout_value;
        auto pid = program_id<scope>();
        auto crd = m_layout.get_hier_coord(pid);
        auto const sl_shape = cute::shape(sl_layout);
        auto const sl_stride = cute::stride(sl_layout);
        auto const m_shape = cute::shape(m_layout);
        int g_dim[t_rank];
        int g_stride[t_rank];
        [&]<size_t... Is>(std::index_sequence<Is...>) {
            ((g_dim[Is] = int(cute::get<Is>(sl_shape)),
              g_stride[Is] = int(cute::get<Is>(sl_stride))),
             ...);
        }(std::make_index_sequence<t_rank>{});
        int m_ext[m_rank], m_crd[m_rank];
        [&]<size_t... Is>(std::index_sequence<Is...>) {
            ((m_ext[Is] = int(cute::get<Is>(m_shape)),
              m_crd[Is] = int(cute::get<Is>(crd))),
             ...);
        }(std::make_index_sequence<m_rank>{});
        int axis_to_mesh[t_rank];
        for (int i = 0; i < t_rank; ++i)
            axis_to_mesh[i] = -1;
        [&]<size_t... Is>(std::index_sequence<Is...>) {
            auto process = [&]<size_t I>(std::integral_constant<size_t, I>) {
                auto attr = cute::get<I>(attrs_t{});
                using A = decltype(attr);
                if constexpr (!std::is_same_v<A, shard::B>) {
                    constexpr int ax = A::axis;
                    axis_to_mesh[ax] = int(I);
                }
            };
            (process(std::integral_constant<size_t, Is>{}), ...);
        }(std::make_index_sequence<m_rank>{});
        int offset = 0;
        for (int i = 0; i < t_rank; ++i) {
            int m = axis_to_mesh[i];
            if (m >= 0) {
                int loc = g_dim[i] >= m_ext[m] ? g_dim[i] / m_ext[m] : g_dim[i];
                offset += m_crd[m] * loc * g_stride[i];
            }
        }
        return offset;
    }
}
