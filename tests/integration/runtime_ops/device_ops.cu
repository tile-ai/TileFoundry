/// Device-side harness for the warp / mbarrier / TMA runtime ops.
///
/// These ops have no codegen emit — their TIR definitions are declared but not
/// wired to the CUDA backend while that chain is rebuilt — so the path
/// test_mma_tir_handwritten.py uses (author TIR, compile, let codegen emit
/// ``tilefoundry::ops::mma``) is not open to them.

/// This file is the substitute: it calls the public entries the way a
/// hand-written kernel does, and the test around it compares to torch. It
/// includes ``runtime.cuh`` rather than the individual ``ops/<op>.cuh``
/// headers, which are included in-context inside ``namespace
/// tilefoundry::ops`` and are not standalone translation units.

#include <tilefoundry/runtime/cuda/runtime.cuh>

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

namespace ops = tilefoundry::ops;

namespace {

constexpr int kThreads = 256;

/// warp.

template <class Combine>
__global__ void warp_reduce_kernel(const float *in, float *out) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    out[t] = ops::warp_reduce<Combine>(in[t]);
}

__global__ void shuffle_xor_kernel(const float *in, float *out, int lane_mask) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    out[t] = ops::shuffle_xor(in[t], lane_mask);
}

/// One counter per block: how many threads the elect admitted. Every entry must
/// read 1, which is the property an arrive-count-1 mbarrier depends on.
template <int Width> __global__ void elect_count_kernel(int *out) {
    if (ops::shuffle_elect<Width>())
        atomicAdd(&out[blockIdx.x], 1);
}

/// mbarrier.

/// One producer thread fills a shared tile and arrives; every consumer waits on
/// the phase. `rounds` forces the parity to flip, which is what lets a fixed
/// ring of barriers serve an unbounded pipeline.
__global__ void mbarrier_pipeline_kernel(const float *in, float *out, int tile,
                                         int rounds) {
    extern __shared__ float sm[];
    __shared__ alignas(8) uint64_t bar;

    if (ops::shuffle_elect<kThreads>())
        ops::mbarrier_init(&bar, 1);
    ops::sync<ops::SyncKind::syncthreads>();

    for (int r = 0; r < rounds; ++r) {
        if (ops::shuffle_elect<kThreads>()) {
            for (int i = 0; i < tile; ++i)
                sm[i] = in[r * tile + i] + float(r);
            ops::mbarrier_arrive(&bar);
        }
        ops::mbarrier_wait_parity(&bar, unsigned(r) & 1u);
        for (int i = threadIdx.x; i < tile; i += kThreads)
            out[r * tile + i] = sm[i];
        ops::sync<ops::SyncKind::syncthreads>();
    }
}

/// tma.

/// A `stages`-deep ring staged by bulk copy, each tile doubled on the way out.
/// The producer declares the byte count on the same instruction it arrives
/// with, so a consumer's parity wait covers the copy without a second barrier.
template <int Stages>
__global__ void tma_stage_kernel(const float *in, float *out, int tile,
                                 int ntile) {
    extern __shared__ __align__(16) float sm[];
    __shared__ alignas(8) uint64_t bar[Stages];

    if (ops::shuffle_elect<kThreads>())
        for (int s = 0; s < Stages; ++s)
            ops::mbarrier_init(&bar[s], 1);
    ops::sync<ops::SyncKind::syncthreads>();

    for (int t = 0; t < ntile; ++t) {
        const int slot = t % Stages;
        const unsigned phase = unsigned(t / Stages) & 1u;
        if (ops::shuffle_elect<kThreads>()) {
            ops::mbarrier_arrive_expect_tx(&bar[slot],
                                           ops::tma_bulk_bytes<float>(tile));
            ops::tma_bulk_copy(in + t * tile, sm + slot * tile, tile,
                               &bar[slot]);
        }
        ops::mbarrier_wait_parity(&bar[slot], phase);
        for (int i = threadIdx.x; i < tile; i += kThreads)
            out[t * tile + i] = sm[slot * tile + i] * 2.f;
        ops::sync<ops::SyncKind::syncthreads>();
    }
}

/// host wrappers.

void check_f32_cuda(const torch::Tensor &t, const char *what) {
    TORCH_CHECK(t.is_cuda() && t.is_contiguous() &&
                    t.scalar_type() == at::kFloat,
                what, " must be a contiguous f32 CUDA tensor");
}

torch::Tensor warp_reduce_sum(torch::Tensor x) {
    check_f32_cuda(x, "x");
    auto out = torch::empty_like(x);
    const int n = int(x.numel());
    TORCH_CHECK(n % kThreads == 0, "x must be a multiple of ", kThreads);
    warp_reduce_kernel<ops::warp_sum>
        <<<n / kThreads, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
            x.data_ptr<float>(), out.data_ptr<float>());
    return out;
}

torch::Tensor warp_reduce_max(torch::Tensor x) {
    check_f32_cuda(x, "x");
    auto out = torch::empty_like(x);
    const int n = int(x.numel());
    TORCH_CHECK(n % kThreads == 0, "x must be a multiple of ", kThreads);
    warp_reduce_kernel<ops::warp_max>
        <<<n / kThreads, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
            x.data_ptr<float>(), out.data_ptr<float>());
    return out;
}

torch::Tensor shuffle_xor(torch::Tensor x, int64_t lane_mask) {
    check_f32_cuda(x, "x");
    auto out = torch::empty_like(x);
    const int n = int(x.numel());
    TORCH_CHECK(n % kThreads == 0, "x must be a multiple of ", kThreads);
    shuffle_xor_kernel<<<n / kThreads, kThreads, 0,
                         at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), int(lane_mask));
    return out;
}

torch::Tensor elect_count(int64_t width, int64_t blocks) {
    auto out = torch::zeros({blocks},
                            torch::dtype(torch::kInt32).device(torch::kCUDA));
    auto stream = at::cuda::getCurrentCUDAStream();
    auto *p = out.data_ptr<int>();
    switch (width) {
    case 8:
        elect_count_kernel<8><<<blocks, kThreads, 0, stream>>>(p);
        break;
    case 32:
        elect_count_kernel<32><<<blocks, kThreads, 0, stream>>>(p);
        break;
    case 256:
        elect_count_kernel<256><<<blocks, kThreads, 0, stream>>>(p);
        break;
    default:
        TORCH_CHECK(false, "elect_count: unsupported width ", width);
    }
    return out;
}

torch::Tensor mbarrier_pipeline(torch::Tensor x, int64_t tile) {
    check_f32_cuda(x, "x");
    const int rounds = int(x.numel() / tile);
    TORCH_CHECK(int64_t(rounds) * tile == x.numel(), "tile must divide x");
    auto out = torch::empty_like(x);
    mbarrier_pipeline_kernel<<<1, kThreads, size_t(tile) * sizeof(float),
                               at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), int(tile), rounds);
    return out;
}

torch::Tensor tma_stage(torch::Tensor x, int64_t tile, int64_t stages) {
    check_f32_cuda(x, "x");
    TORCH_CHECK(tile * sizeof(float) % 16 == 0,
                "tma_bulk_copy needs a 16-byte multiple; tile=", tile);
    const int ntile = int(x.numel() / tile);
    TORCH_CHECK(int64_t(ntile) * tile == x.numel(), "tile must divide x");
    auto out = torch::empty_like(x);
    const size_t smem = size_t(tile) * size_t(stages) * sizeof(float);
    auto stream = at::cuda::getCurrentCUDAStream();
    switch (stages) {
    case 1:
        tma_stage_kernel<1><<<1, kThreads, smem, stream>>>(
            x.data_ptr<float>(), out.data_ptr<float>(), int(tile), ntile);
        break;
    case 3:
        tma_stage_kernel<3><<<1, kThreads, smem, stream>>>(
            x.data_ptr<float>(), out.data_ptr<float>(), int(tile), ntile);
        break;
    default:
        TORCH_CHECK(false, "tma_stage: unsupported stage count ", stages);
    }
    return out;
}

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("warp_reduce_sum", &warp_reduce_sum);
    m.def("warp_reduce_max", &warp_reduce_max);
    m.def("shuffle_xor", &shuffle_xor);
    m.def("elect_count", &elect_count);
    m.def("mbarrier_pipeline", &mbarrier_pipeline);
    m.def("tma_stage", &tma_stage);
}
