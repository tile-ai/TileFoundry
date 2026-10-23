/// CUDA copy op implementation. Included in-context from ops/copy.cuh inside
/// namespace tilefoundry::ops.
#pragma once

namespace copy_impl {

/// The slice of a plain operand that pairs with one shard operand.
///
/// An operand with no shard layout of its own is not therefore whole for
/// every instance: when the other operand is sharded, the plain side moves
/// exactly the slice that instance owns, projected through the shard side's
/// layout, whose shape refines the shared logical domain. The projection's
/// window and origin are row-major logical steps, and the plain layout is
/// reversed before they index it, so CuTe's first-mode-fastest decode reads
/// them as the row-major index they are.
template <class T, class L, class A, class M>
CUTE_HOST_DEVICE auto local_as(T &&t,
                               tilefoundry::ShardLayout<L, A, M> const &sl) {
    auto logical =
        cute::make_layout(cute::shape(sl.layout_value), cute::GenRowMajor{});
    tilefoundry::ShardLayout<decltype(logical), A, M> as_logical{logical,
                                                                 sl.mesh_value};
    auto const [window, origin] = tilefoundry::detail::local_layout_and_offset(
        as_logical,
        tilefoundry::mesh_coords(sl.mesh_value, tilefoundry::program_ids()));
    auto const plain = reverse(cute::flatten(t.layout()));
    return cute::make_tensor(t.data() + plain(origin),
                             cute::composition(plain, window));
}

/// ``copy`` is ``cute::copy`` with the operands resolved to this slice first.
struct Copy {
    template <class TSrc, class TDst>
    __device__ void operator()(TSrc const &src, TDst &dst) const {
        if constexpr (tilefoundry::ShardTensorLike<TSrc> &&
                      !tilefoundry::ShardTensorLike<TDst>) {
            auto &&s = tilefoundry::local_tensor(src);
            auto d = local_as(dst, src.shard_layout);
            cute::copy(s, d);
        } else if constexpr (!tilefoundry::ShardTensorLike<TSrc> &&
                             tilefoundry::ShardTensorLike<TDst>) {
            auto s = local_as(src, dst.shard_layout);
            auto &&d = tilefoundry::local_tensor(dst);
            cute::copy(s, d);
        } else {
            auto &&s = tilefoundry::local_tensor(src);
            auto &&d = tilefoundry::local_tensor(dst);
            cute::copy(s, d);
        }
    }
};

/// Whether two projected views hold the same number of elements.
template <class SView, class DView>
CUTE_HOST_DEVICE constexpr bool same_slice_size() {
    using SL = typename cute::remove_cvref_t<SView>::layout_type;
    using DL = typename cute::remove_cvref_t<DView>::layout_type;
    if constexpr (!cute::is_static<SL>::value || !cute::is_static<DL>::value)
        return true;
    else
        return int(decltype(cute::size(SL{}))::value) ==
               int(decltype(cute::size(DL{}))::value);
}

/// Bytes in the largest common asynchronous move.
template <class SView, class DView>
CUTE_HOST_DEVICE constexpr int async_bytes() {
    using SL = typename cute::remove_cvref_t<SView>::layout_type;
    using DL = typename cute::remove_cvref_t<DView>::layout_type;
    using elem = typename cute::remove_cvref_t<DView>::value_type;
    if constexpr (!cute::is_static<SL>::value || !cute::is_static<DL>::value) {
        return int(sizeof(elem));
    } else {
        constexpr int run = int(
            decltype(cute::gcd(cute::max_common_vector(SL{}, DL{}),
                               cute::gcd(cute::max_alignment(SL{}),
                                         cute::max_alignment(DL{}))))::value);
        int b = int(sizeof(elem));
        while (b * 2 <= run * int(sizeof(elem)) && b * 2 <= 16)
            b *= 2;
        return b;
    }
}

/// The same move, issued asynchronously.
struct CopyAsync {
    template <class TSrc, class TDst>
    __device__ void operator()(TSrc const &src, TDst &dst) const {
        auto &&s = tilefoundry::local_tensor(src);
        auto &&d = tilefoundry::local_tensor(dst);
        using value_type =
            typename cute::remove_cvref_t<decltype(d)>::value_type;
        constexpr int bytes = async_bytes<decltype(s), decltype(d)>();
        static_assert(bytes > int(sizeof(value_type)),
                      "ops::copy_async: these two views share no run wide "
                      "enough for cp.async");
        static_assert(
            same_slice_size<decltype(s), decltype(d)>(),
            "ops::copy_async: the two projected slices must hold the same "
            "number of elements");
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 800)
        auto s_v = cute::recast<cute::uint_bit_t<bytes * 8>>(s);
        auto d_v = cute::recast<cute::uint_bit_t<bytes * 8>>(d);
        const int nv = int(cute::size(s_v));
        for (int i = 0; i < nv; ++i)
            __pipeline_memcpy_async(&d_v(i), &s_v(i), bytes);
#else
        static_assert(dependent_false_v<TSrc>,
                      "ops::copy_async: cp.async requires sm_80 or newer");
#endif
    }
};

/// Gather complete rows, distributing destination vectors over the issuer mesh.
template <class ExecutionMesh, bool HasFill, int bytes>
struct CopyIndexedAsync {
    template <class TSrc, class TDst, class TIndex, class TFill>
    __device__ void operator()(TSrc const &src, TDst &dst, TIndex const &index,
                               TFill fill) const {
        auto &&s = tilefoundry::local_tensor(src);
        auto &&d = tilefoundry::local_tensor(dst);
        auto &&idx = tilefoundry::local_tensor(index);
        constexpr int srank = decltype(cute::rank(s))::value;
        constexpr int drank = decltype(cute::rank(d))::value;
        static_assert(srank >= 2 && drank == srank,
                      "indexed copy_async requires a row with inner modes");
        auto rows_s = cute::group_modes<1, srank>(s);
        auto rows_d = cute::group_modes<1, drank>(d);
        auto row_s = rows_s(0, cute::_);
        auto row_d = rows_d(0, cute::_);
        using value_type =
            typename cute::remove_cvref_t<decltype(d)>::value_type;
        static_assert(
            bytes == 4 || bytes == 8 || bytes == 16,
            "indexed copy_async needs an aligned 4, 8, or 16 byte row vector");
        constexpr int V = bytes / int(sizeof(value_type));
        constexpr auto threads =
            tilefoundry::get<tilefoundry::TopologyScope::thread>(
                ExecutionMesh{});
        constexpr int participants = int(cute::size(threads.layout));
        const int lane = int(threadIdx.x) - tilefoundry::offset(threads);
        auto row_vectors = cute::recast<cute::uint_bit_t<bytes * 8>>(row_d);
        const int vectors_per_row = int(cute::size(row_vectors));
        const int vectors = int(cute::size<0>(rows_d)) * vectors_per_row;
        for (int vector = lane; vector < vectors; vector += participants) {
            const int row = vector / vectors_per_row;
            const int column = vector % vectors_per_row;
            const auto at = idx(row);
            const bool valid = at >= 0 && at < cute::size<0>(rows_s);
            auto destination = rows_d(row, cute::_);
            auto destination_vectors =
                cute::recast<cute::uint_bit_t<bytes * 8>>(destination);
            auto *target = &destination_vectors(column);
            if (!HasFill || valid || fill == TFill(0)) {
                auto selected = rows_s(valid ? at : 0, cute::_);
                auto selected_vectors =
                    cute::recast<cute::uint_bit_t<bytes * 8>>(selected);
                const auto *source = &selected_vectors(column);
                const int source_bytes = !HasFill || valid ? bytes : 0;
                const uint32_t shared =
                    uint32_t(__cvta_generic_to_shared(target));
                if constexpr (bytes == 16) {
                    asm volatile(
                        "cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(
                            shared),
                        "l"(source), "r"(source_bytes)
                        : "memory");
                } else {
                    asm volatile(
                        "cp.async.ca.shared.global [%0], [%1], %2, %3;" ::"r"(
                            shared),
                        "l"(source), "n"(bytes), "r"(source_bytes)
                        : "memory");
                }
            } else {
                alignas(16) value_type pattern[V];
                CUTE_UNROLL
                for (int element = 0; element < V; ++element)
                    pattern[element] = value_type(fill);
                const auto *words = reinterpret_cast<uint32_t const *>(pattern);
                const uint32_t shared =
                    uint32_t(__cvta_generic_to_shared(target));
                if constexpr (bytes == 16) {
                    asm volatile(
                        "st.shared.v4.b32 [%0], {%1, %2, %3, %4};" ::"r"(
                            shared),
                        "r"(words[0]), "r"(words[1]), "r"(words[2]),
                        "r"(words[3])
                        : "memory");
                } else if constexpr (bytes == 8) {
                    asm volatile(
                        "st.shared.v2.b32 [%0], {%1, %2};" ::"r"(shared),
                        "r"(words[0]), "r"(words[1])
                        : "memory");
                } else {
                    asm volatile("st.shared.b32 [%0], %1;" ::"r"(shared),
                                 "r"(words[0])
                                 : "memory");
                }
            }
        }
    }
};

}
