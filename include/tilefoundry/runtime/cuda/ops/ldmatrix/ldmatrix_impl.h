/// CUDA ldmatrix implementation. Included in-context from ops/ldmatrix.cuh
/// inside namespace tilefoundry::ops.
#pragma once

namespace ldmatrix_impl {

__device__ inline uint32_t smem_addr(void const *ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

template <class Src>
__device__ decltype(auto) source_at(Src const &src, int row, int col) {
    if constexpr (tilefoundry::ShardTensorLike<Src>) {
        auto whole =
            cute::make_tensor(src.data(), src.shard_layout.layout_value);
        return whole(row, col);
    } else {
        return src(row, col);
    }
}

struct LdMatrix {
    template <class Src, class Dst>
    __device__ void operator()(Src const &src, Dst &dst) const {
        auto d = dst.data();
        const int lane =
            int(tilefoundry::program_id<tilefoundry::TopologyScope::thread>()) &
            31;
        const int row = lane & 15;
        const int col = (lane >> 4) * 8;
        uint32_t r0, r1, r2, r3;
        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
            : "r"(smem_addr(&source_at(src, row, col))));

        /// Mma's A fragment is stored in layout order. Its runtime packs that
        /// order as [0, 1], [4, 5], [2, 3], [6, 7], so place the four PTX
        /// registers where that packing reads them back in r0..r3 order.
        __builtin_memcpy(&d[0], &r0, sizeof(r0));
        __builtin_memcpy(&d[4], &r1, sizeof(r1));
        __builtin_memcpy(&d[2], &r2, sizeof(r2));
        __builtin_memcpy(&d[6], &r3, sizeof(r3));
    }
};

}
