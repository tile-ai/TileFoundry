/// Warp-level hardware primitives.
#pragma once

namespace warp_util {

/// Whether __shfl_xor_sync accepts T directly.
template <class T>
inline constexpr bool is_shuffle_native_v =
    std::is_same_v<T, float> || std::is_same_v<T, double> ||
    std::is_same_v<T, int> || std::is_same_v<T, unsigned int> ||
    std::is_same_v<T, long long> || std::is_same_v<T, unsigned long long>;

/// Shuffle native values directly and aggregates word by word.
template <class T> struct ShuffleXor {
    __device__ T operator()(T value, int lane_mask,
                            unsigned member_mask) const {
        if constexpr (is_shuffle_native_v<T>) {
            return __shfl_xor_sync(member_mask, value, lane_mask);
        } else {
            static_assert(
                sizeof(T) % sizeof(unsigned) == 0,
                "shuffle_xor: type size must be a multiple of 4 bytes");
            constexpr int kWords = int(sizeof(T) / sizeof(unsigned));
            T out = value;
            unsigned *dst = reinterpret_cast<unsigned *>(&out);
            for (int i = 0; i < kWords; ++i)
                dst[i] = __shfl_xor_sync(member_mask, dst[i], lane_mask);
            return out;
        }
    }
};

/// Elect exactly one thread of the CTA.
struct Elect {
    __device__ bool operator()() const {
        const size_t tid = program_id<TopologyScope::thread>();
        if (tid >= size_t(kWarpSize))
            return false;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
        constexpr unsigned kWholeWarp = 0xFFFFFFFFu;
        unsigned pred = 0u;
        asm volatile("{\n"
                     "  .reg .b32 elected_lane;\n"
                     "  .reg .pred is_elected;\n"
                     "  elect.sync elected_lane|is_elected, %1;\n"
                     "  selp.b32 %0, 1, 0, is_elected;\n"
                     "}\n"
                     : "=r"(pred)
                     : "n"(kWholeWarp));
        return pred != 0u;
#else
        return tid == 0;
#endif
    }
};

}

/// Exchange ``value`` with the lane whose id differs in ``lane_mask``.
template <class T>
__device__ inline T shuffle_xor(T value, int lane_mask,
                                unsigned member_mask = 0xFFFFFFFFu) {
    return warp_util::ShuffleXor<T>{}(value, lane_mask, member_mask);
}

/// One thread of the CTA answers true; every other answers false.
__device__ inline bool shuffle_elect() { return warp_util::Elect{}(); }

/// Fold ``value`` across ``Width`` lanes, every lane left holding the total.
template <class Combine, int Width = 32, class T>
__device__ inline T warp_reduce(T value) {
    static_assert(Width >= 2 && Width <= 32 && (Width & (Width - 1)) == 0,
                  "warp_reduce: Width must be a power of two up to 32");
    for (int delta = Width >> 1; delta > 0; delta >>= 1)
        value = Combine{}(value, shuffle_xor(value, delta));
    return value;
}
