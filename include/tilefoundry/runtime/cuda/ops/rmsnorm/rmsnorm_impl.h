/// CUDA RMSNorm op implementation. Included in-context from ops/rmsnorm.cuh
/// inside namespace tilefoundry::ops.
#pragma once

namespace rmsnorm_impl {

struct RmsNorm {
    template <class TIn, class TOut, class TW>
    __device__ void operator()(TIn const &src, TOut &dst, TW const &weight,
                               int M, int K, float eps) const {
        auto s = detail::to_local(src);
        auto &&d = detail::to_local(dst);
        auto w = detail::to_local(weight);
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
