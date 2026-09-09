/// Compile-time view of the warp axes of a thread mesh.
#pragma once

namespace detail {

template <class L, bool = cute::is_composed_layout<L>::value>
struct mesh_warp_positions {
    using type = L;
};
template <class L> struct mesh_warp_positions<L, true> {
    using type = cute::remove_cvref_t<decltype(std::declval<L>().layout_b())>;
};

template <class L, int Rank> struct mesh_warp_geometry {
    static constexpr int lane_extent = kWarpSize;
    static constexpr int warp_stride = 1;
};
template <class L> struct mesh_warp_geometry<L, 1> {
    static constexpr int lane_extent = int(cute::get<0>(cute::shape(L{})));
    static constexpr int warp_stride = 1;
};
template <class L> struct mesh_warp_geometry<L, 2> {
    static constexpr int lane_extent = int(cute::get<1>(cute::shape(L{})));
    static constexpr int warp_stride =
        int(cute::get<0>(cute::stride(L{}))) / kWarpSize;
};

template <class TMesh> struct MeshWarpView {
    using layout_t = typename TMesh::layout;
    using positions_t = typename mesh_warp_positions<layout_t>::type;
    static constexpr int rank = decltype(cute::rank(positions_t{}))::value;
    static constexpr int raw_lane_extent =
        mesh_warp_geometry<positions_t, rank>::lane_extent;
    static constexpr int lane_extent =
        raw_lane_extent < kWarpSize ? raw_lane_extent : kWarpSize;
    static constexpr int lane_stride = 1;
    static constexpr int warp_count =
        int(cute::size(positions_t{})) / kWarpSize;
    static constexpr int warp_stride =
        mesh_warp_geometry<positions_t, rank>::warp_stride;
    static constexpr int first_warp =
        mesh_layout_offset<cute::remove_cvref_t<layout_t>>::value / kWarpSize;
    using lane_layout = cute::Layout<cute::Shape<cute::Int<lane_extent>>,
                                     cute::Stride<cute::Int<lane_stride>>>;
    using warp_layout = cute::Layout<cute::Shape<cute::Int<warp_count>>,
                                     cute::Stride<cute::Int<warp_stride>>>;
    static constexpr bool whole_warps = lane_extent == kWarpSize;
    static constexpr bool warps_contiguous() { return warp_stride == 1; }
    static constexpr int warps() { return warp_count; }
};

}
