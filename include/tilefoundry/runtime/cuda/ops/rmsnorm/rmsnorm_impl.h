/// CUDA RMSNorm op implementation. Included in-context from ops/rmsnorm.cuh
/// inside namespace tilefoundry::ops.
#pragma once

namespace rmsnorm_impl {

struct RmsNorm {
    template <class TIn, class TOut, class TW>
    __device__ void operator()(TIn const &src, TOut &dst, TW const &weight,
                               float eps) const {
        auto s = detail::to_local(src);
        auto &&d = detail::to_local(dst);
        auto w = detail::to_local(weight);

        /// Normalize each row using shard-layout M and K.
        using dst_type = cute::remove_cvref_t<TOut>;
        using src_type = cute::remove_cvref_t<TIn>;
        using weight_type = cute::remove_cvref_t<TW>;
        using dst_layout = typename dst_type::shard_layout_type::layout;
        using src_layout = typename src_type::shard_layout_type::layout;
        using weight_layout = typename weight_type::shard_layout_type::layout;
        static_assert(decltype(cute::rank(dst_layout{}))::value == 2,
                      "ops::rmsnorm: destination shard layout must be rank 2");
        constexpr int M = int(cute::size<0>(dst_layout{}));
        constexpr int K = int(cute::size<1>(dst_layout{}));
        static_assert(decltype(cute::rank(src_layout{}))::value == 2,
                      "ops::rmsnorm: source shard layout must be rank 2");
        static_assert(tilefoundry::detail::shard_layout_is_full_broadcast<
                          typename dst_type::shard_layout_type>(),
                      "ops::rmsnorm: destination must hold the whole tile");
        static_assert(
            tilefoundry::detail::shard_layout_is_full_broadcast<
                typename src_type::shard_layout_type>(),
            "ops::rmsnorm: source must hold the whole tile because the loops "
            "index its projected view with the destination's M and K");
        static_assert(decltype(cute::rank(weight_layout{}))::value == 1,
                      "ops::rmsnorm: weight shard layout must be rank 1");
        static_assert(int(cute::size<0>(weight_layout{})) == K,
                      "ops::rmsnorm: weight must be a vector of length K");
        using dst_view = tilefoundry::detail::local_view_t<TOut>;
        using src_view = tilefoundry::detail::local_view_t<TIn>;
        static_assert(
            decltype(cute::rank(typename dst_view::layout_type{}))::value ==
                    1 &&
                decltype(cute::rank(typename src_view::layout_type{}))::value ==
                    1,
            "ops::rmsnorm: projected source and destination must be rank 1");
        static_assert(
            int(cute::size(typename dst_view::layout_type{})) == M * K &&
                int(cute::size(typename src_view::layout_type{})) == M * K,
            "ops::rmsnorm: projected views must hold exactly M * K elements");

        using value_type = cute::remove_cvref_t<decltype(d(0))>;
        for (int m = 0; m < M; ++m) {
            float sum_sq = 0.0f;
            for (int k = 0; k < K; ++k) {
                float val = static_cast<float>(s(m * K + k));
                sum_sq += val * val;
            }
            float rms = rsqrtf(sum_sq / float(K) + eps);
            for (int k = 0; k < K; ++k) {
                float val = static_cast<float>(s(m * K + k)) * rms *
                            static_cast<float>(w(k));
                d(m * K + k) = static_cast<value_type>(val);
            }
        }
    }
};

}
