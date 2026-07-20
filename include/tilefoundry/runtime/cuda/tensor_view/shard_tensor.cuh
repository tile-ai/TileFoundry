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

// Shared shard-projection state: mesh coordinate + per-axis global extents /
// strides + the mesh-axis-to-tensor-axis mapping, derived once from a
// ShardTensor's ``ShardLayout``. Consumed by both ``local_impl`` (the
// pointer-offset projection behind ``local()``) and ``local_offset`` (a
// distinct, narrower index computation — see the comment there) so the
// ~45-line preamble (type extraction, ``program_id``/``get_hier_coord``,
// shape/stride unpack, attr-to-axis fold) exists exactly once.
template <int TRank, int MRank> struct shard_axis_projection_t {
    static constexpr int t_rank = TRank;
    static constexpr int m_rank = MRank;
    int g_dim[TRank];
    int g_stride[TRank];
    int m_ext[MRank];
    int m_crd[MRank];
    int axis_to_mesh[TRank]; // -1 when tensor axis i is not a Split position
};

template <class T, class GL, class SL>
CUTE_HOST_DEVICE auto shard_axis_projection(ShardTensor<T, GL, SL> const &st) {
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

    shard_axis_projection_t<t_rank, m_rank> proj{};
    [&]<size_t... Is>(std::index_sequence<Is...>) {
        ((proj.g_dim[Is] = int(cute::get<Is>(sl_shape)),
          proj.g_stride[Is] = int(cute::get<Is>(sl_stride))),
         ...);
    }(std::make_index_sequence<t_rank>{});

    [&]<size_t... Is>(std::index_sequence<Is...>) {
        ((proj.m_ext[Is] = int(cute::get<Is>(m_shape)),
          proj.m_crd[Is] = int(cute::get<Is>(crd))),
         ...);
    }(std::make_index_sequence<m_rank>{});

    for (int i = 0; i < t_rank; ++i)
        proj.axis_to_mesh[i] = -1;
    [&]<size_t... Is>(std::index_sequence<Is...>) {
        auto process = [&]<size_t I>(std::integral_constant<size_t, I>) {
            auto attr = cute::get<I>(attrs_t{});
            using A = decltype(attr);
            if constexpr (!std::is_same_v<A, shard::B>) {
                constexpr int ax = A::axis;
                proj.axis_to_mesh[ax] = int(I);
            }
        };
        (process(std::integral_constant<size_t, Is>{}), ...);
    }(std::make_index_sequence<m_rank>{});

    return proj;
}

// Full-broadcast predicate shared by ``local()`` and ``local_offset()``: a
// ShardLayout with no attrs, or with fewer attrs than mesh axes, carries no
// per-axis Split contribution at all.
template <class SL>
CUTE_HOST_DEVICE constexpr bool shard_layout_is_full_broadcast() {
    using attrs_t = typename SL::attrs;
    using mesh_t = typename SL::mesh;
    using m_layout_t = typename mesh_t::layout;
    using m_shape_t = cute::remove_cvref_t<decltype(cute::shape(m_layout_t{}))>;
    return cute::tuple_size<attrs_t>::value == 0 ||
           cute::tuple_size<attrs_t>::value !=
               cute::tuple_size<m_shape_t>::value;
}

template <class T, class GL, class SL>
CUTE_HOST_DEVICE auto local_impl(ShardTensor<T, GL, SL> const &st,
                                 std::true_type /*full_broadcast*/) {
    return st.engine;
}

template <class T, class GL, class SL>
CUTE_HOST_DEVICE auto local_impl(ShardTensor<T, GL, SL> const &st,
                                 std::false_type /*full_broadcast*/) {
    auto const proj = shard_axis_projection(st);
    constexpr int t_rank = decltype(proj)::t_rank;

    // spec runtime.md §2.10.2: offset = Σ_{m : A[m]=Split(k)} coord[m]·S[k].
    // Valid under the §7.1.1 canonical-layout precondition (every Split
    // position's own extent already equals its mesh axis's extent); see
    // ``local_offset`` below for a distinct, loc-scaled quantity.
    int loc_shape[t_rank];
    int loc_stride[t_rank];
    int offset = 0;
    for (int i = 0; i < t_rank; ++i) {
        int m = proj.axis_to_mesh[i];
        if (m >= 0) {
            offset += proj.m_crd[m] * proj.g_stride[i];
            loc_shape[i] = proj.g_dim[i] / proj.m_ext[m];
            loc_stride[i] = proj.g_stride[i];
        } else {
            loc_shape[i] = proj.g_dim[i];
            loc_stride[i] = proj.g_stride[i];
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
        constexpr bool full_bc = shard_layout_is_full_broadcast<SL>();
        return local_impl(st, std::bool_constant<full_bc>{});
    }
}

// Local INDEX offset (not a global memory-pointer offset) into an
// already-projected destination view — consumed only by
// ``ops::copy_async``'s partial-broadcast destination indexing
// (ops/copy/copy_impl.h), where the destination's local view spans more
// elements than the source's and each thread's slice must land at its own
// sub-range.
//
// Unlike ``local_impl``'s pointer offset above (spec §2.10.2's
// Σ coord[m]·S[k]), this additionally scales by
// ``loc = g_dim[i]/m_ext[m]`` (clamped to ``g_dim[i]`` when smaller) before
// multiplying by the stride. Under the §7.1.1 canonical-layout precondition
// every Split position's extent already equals its mesh axis's extent, so
// ``loc`` is always 1 there and the two formulas coincide; the extra factor
// only has effect for a Split position whose own extent exceeds its mesh
// axis's extent (a shape the §7.1.1-canonical form should not produce — see
// repo-dedup-and-test-trim.findings.md F28/F43). This divergence predates
// this refactor and is intentionally left as-is here (documented, not
// resolved) rather than risk a silent behavior change.
template <class T, class GL, class SL>
CUTE_HOST_DEVICE int local_offset(ShardTensor<T, GL, SL> const &st) {
    if constexpr (shard_layout_is_full_broadcast<SL>()) {
        return 0;
    } else {
        auto const proj = shard_axis_projection(st);
        constexpr int t_rank = decltype(proj)::t_rank;
        int offset = 0;
        for (int i = 0; i < t_rank; ++i) {
            int m = proj.axis_to_mesh[i];
            if (m >= 0) {
                int loc = proj.g_dim[i] >= proj.m_ext[m]
                              ? proj.g_dim[i] / proj.m_ext[m]
                              : proj.g_dim[i];
                offset += proj.m_crd[m] * loc * proj.g_stride[i];
            }
        }
        return offset;
    }
}
