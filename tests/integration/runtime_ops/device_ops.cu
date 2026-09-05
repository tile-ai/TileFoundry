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
#include <tuple>
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

/// copy.

/// A run of `N` as a ShardTensor over a `Mesh` of `Threads` instances.
///
/// A split run is spelled `(Threads, N/Threads)` with the mesh on the leading
/// axis, which is the canonical Split position [runtime
/// §2.10.2](docs/spec/runtime.md#2102-computation) asks for: `local()` offsets
/// by `coord * stride`, so the per-instance block has to *be* that stride.
/// `Stride` then says whether the run is contiguous, which is what decides the
/// width.
template <int N, int Stride, int Threads, class Ptr>
__device__ auto run(Ptr p) {
    auto mesh_layout = cute::make_layout(cute::make_shape(cute::Int<Threads>{}),
                                         cute::make_stride(cute::Int<1>{}));
    tilefoundry::Mesh<
        tilefoundry::Topology<tilefoundry::TopologyScope::thread, Threads>,
        decltype(mesh_layout)>
        mesh{mesh_layout};
    if constexpr (Threads == 1) {
        auto layout = cute::make_layout(cute::make_shape(cute::Int<N>{}),
                                        cute::make_stride(cute::Int<Stride>{}));
        tilefoundry::ShardLayout<decltype(layout),
                                 cute::tuple<tilefoundry::shard::B>,
                                 decltype(mesh)>
            shard{layout, mesh};
        return tilefoundry::make_shard_tensor(cute::make_tensor(p, layout),
                                              layout, shard);
    } else {
        constexpr int per = N / Threads;
        auto layout = cute::make_layout(
            cute::make_shape(cute::Int<Threads>{}, cute::Int<per>{}),
            cute::make_stride(cute::Int<Stride * per>{}, cute::Int<Stride>{}));
        tilefoundry::ShardLayout<decltype(layout),
                                 cute::tuple<tilefoundry::shard::S<0>>,
                                 decltype(mesh)>
            shard{layout, mesh};
        return tilefoundry::make_shard_tensor(cute::make_tensor(p, layout),
                                              layout, shard);
    }
}

/// Stage a gmem run into shared with `ops::copy`, then hand it back.
///
/// `Threads` says how the mesh splits the run, and that is the only place a
/// thread index appears: the call itself names neither a share nor a width.
/// `Want` is what the layouts should have chosen, checked at compile time. A
/// mesh smaller than the block would leave the surplus threads without a
/// coordinate, so a split mesh here is the whole block.
template <int N, int Stride, int Threads, int Want, class T>
__global__ void copy_kernel(const T *in, T *out) {
    __shared__ __align__(16) T sm[N];
    auto src = run<N, Stride, Threads>(cute::make_gmem_ptr(in));
    auto dst = run<N, 1, Threads>(cute::make_smem_ptr(sm));
    static_assert(ops::copy_vector_elems<decltype(src), decltype(dst)> == Want,
                  "the move width must follow the two shard layouts");
    if constexpr (Threads == 1) {
        if (ops::shuffle_elect<kThreads>())
            ops::copy(src, dst);
    } else {
        ops::copy(src, dst);
    }
    ops::sync<ops::SyncKind::syncthreads>();
    for (int i = threadIdx.x; i < N; i += kThreads)
        out[i] = sm[i];
}

/// A dtype change on the way through, so the width follows the wider side.
template <int N, int Want>
__global__ void copy_widen_kernel(const __nv_bfloat16 *in, float *out) {
    __shared__ __align__(16) float sm[N];
    auto src = run<N, 1, 1>(cute::make_gmem_ptr(in));
    auto dst = run<N, 1, 1>(cute::make_smem_ptr(sm));
    static_assert(ops::copy_vector_elems<decltype(src), decltype(dst)> == Want,
                  "a dtype change caps the width at the wider element");
    if (ops::shuffle_elect<kThreads>())
        ops::copy(src, dst);
    ops::sync<ops::SyncKind::syncthreads>();
    for (int i = threadIdx.x; i < N; i += kThreads)
        out[i] = sm[i];
}

/// mma.

/// A rank-2 tile as a ShardTensor over a one-instance mesh.
///
/// The strides are the whole transpose story: a `(N, K)` view with stride
/// `(K, 1)` is a k-major buffer and one with `(1, N)` is an n-major one, and
/// `ops::mma` reads both through the same indexing.
template <int R, int C, int S0, int S1, class Ptr> __device__ auto tile(Ptr p) {
    auto layout =
        cute::make_layout(cute::make_shape(cute::Int<R>{}, cute::Int<C>{}),
                          cute::make_stride(cute::Int<S0>{}, cute::Int<S1>{}));
    auto mesh_layout = cute::make_layout(cute::make_shape(cute::Int<1>{}),
                                         cute::make_stride(cute::Int<1>{}));
    tilefoundry::Mesh<tilefoundry::Topology<tilefoundry::TopologyScope::cta, 1>,
                      decltype(mesh_layout)>
        mesh{mesh_layout};
    tilefoundry::ShardLayout<decltype(layout),
                             cute::tuple<tilefoundry::shard::B>, decltype(mesh)>
        shard{layout, mesh};
    return tilefoundry::make_shard_tensor(cute::make_tensor(p, layout), layout,
                                          shard);
}

/// `acc = a @ b` over a whole `M x N x K` tile, with `b` given as `(N, K)`.
///
/// `BKMajor` picks which buffer `b` names -- a `(N, K)` one or a `(K, N)` one
/// -- and nothing but the strides changes, which is the point: `ops::mma` has
/// no transpose flag to set.
template <int M, int N, int K, bool BKMajor>
__global__ void mma_tile_kernel(const __nv_bfloat16 *ga,
                                const __nv_bfloat16 *gb, float *gc) {
    __shared__ __align__(16) __nv_bfloat16 sa[M * K];
    __shared__ __align__(16) __nv_bfloat16 sb[N * K];
    for (int i = threadIdx.x; i < M * K; i += kThreads)
        sa[i] = ga[i];
    for (int i = threadIdx.x; i < N * K; i += kThreads)
        sb[i] = gb[i];
    ops::sync<ops::SyncKind::syncthreads>();

    constexpr int elems = ops::mma_acc_elems<M, N, kThreads>();
    float acc[elems];
    for (int i = 0; i < elems; ++i)
        acc[i] = 0.f;
    auto c = ops::mma_acc_tensor<M, N, kThreads>(cute::make_rmem_ptr(&acc[0]));

    auto a = tile<M, K, K, 1>(cute::make_smem_ptr(sa));
    auto run = [&](auto b) {
        static_assert(ops::mma_is_tile<decltype(a), decltype(b), decltype(c)>,
                      "rank-2 static operands must reach the tile tier");
        ops::mma(a, b, c);
    };
    if constexpr (BKMajor) {
        run(tile<N, K, K, 1>(cute::make_smem_ptr(sb)));
    } else {
        run(tile<N, K, 1, N>(cute::make_smem_ptr(sb)));
    }

    for (int f = 0; f < elems; ++f) {
        auto rc = ops::mma_acc_coord<M, N, kThreads>(f, int(threadIdx.x));
        gc[cute::get<0>(rc) * N + cute::get<1>(rc)] = acc[f];
    }
}

/// tma.

/// A `Stages`-deep ring staged by `ops::tma_copy`, each tile doubled on the way
/// out. `SrcStride` selects the tier without changing the call.
template <int Tile, int Stages, int SrcStride>
__global__ void tma_stage_kernel(const float *in, float *out, int ntile) {
    __shared__ __align__(16) float sm[Tile * Stages];
    __shared__ alignas(8) uint64_t bar[Stages];

    if (ops::shuffle_elect<kThreads>())
        for (int s = 0; s < Stages; ++s)
            ops::mbarrier_init(&bar[s], 1);
    ops::sync<ops::SyncKind::syncthreads>();

    for (int t = 0; t < ntile; ++t) {
        const int slot = t % Stages;
        const unsigned phase = unsigned(t / Stages) & 1u;
        auto src = run<Tile, SrcStride, 1>(
            cute::make_gmem_ptr(in + size_t(t) * Tile * SrcStride));
        auto dst = run<Tile, 1, 1>(cute::make_smem_ptr(sm + slot * Tile));
        static_assert(ops::tma_copy_is_bulk<decltype(src), decltype(dst)> ==
                          (SrcStride == 1),
                      "the tier must follow the source layout's stride");
        ops::tma_copy(src, dst, &bar[slot]);
        ops::mbarrier_wait_parity(&bar[slot], phase);
        for (int i = threadIdx.x; i < Tile; i += kThreads)
            out[t * Tile + i] = sm[slot * Tile + i] * 2.f;
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

/// `tile` picks the transfer shape, `src_stride` the tier: 1024 floats at
/// stride 1 is the bulk instruction, at stride 2 the element path, and 6 floats
/// is 24 bytes -- contiguous, so bulk-eligible, but off the 16-byte grain, so
/// the entry hands it to the element path at run time. All three are the same
/// `ops::tma_copy` call and must land the same values.
torch::Tensor mma_tile(torch::Tensor a, torch::Tensor b, bool b_k_major) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "a and b must be CUDA tensors");
    constexpr int M = 16, N = 128, K = 128;
    TORCH_CHECK(a.numel() == M * K && b.numel() == N * K,
                "mma_tile is instantiated for 16x128x128");
    auto ab = a.to(at::kBFloat16).contiguous();
    auto bb = b.to(at::kBFloat16).contiguous();
    auto out =
        torch::empty({M, N}, torch::dtype(torch::kFloat32).device(a.device()));
    auto stream = at::cuda::getCurrentCUDAStream();
    auto *pa = reinterpret_cast<const __nv_bfloat16 *>(ab.data_ptr());
    auto *pb = reinterpret_cast<const __nv_bfloat16 *>(bb.data_ptr());
    if (b_k_major)
        mma_tile_kernel<M, N, K, true>
            <<<1, kThreads, 0, stream>>>(pa, pb, out.data_ptr<float>());
    else
        mma_tile_kernel<M, N, K, false>
            <<<1, kThreads, 0, stream>>>(pa, pb, out.data_ptr<float>());
    return out;
}

torch::Tensor copy_tile(torch::Tensor x, int64_t src_stride, int64_t threads) {
    check_f32_cuda(x, "x");
    auto in = x.to(at::kBFloat16);
    auto out = torch::empty({x.numel() / src_stride}, in.options());
    auto stream = at::cuda::getCurrentCUDAStream();
    auto *p = reinterpret_cast<const __nv_bfloat16 *>(in.data_ptr());
    auto *o = reinterpret_cast<__nv_bfloat16 *>(out.data_ptr());
    const auto key = std::make_tuple(out.numel(), src_stride, threads);
    if (key == std::make_tuple(int64_t(1024), int64_t(1), int64_t(1)))
        copy_kernel<1024, 1, 1, 8><<<1, kThreads, 0, stream>>>(p, o);
    else if (key == std::make_tuple(int64_t(1024), int64_t(2), int64_t(1)))
        copy_kernel<1024, 2, 1, 1><<<1, kThreads, 0, stream>>>(p, o);
    else if (key == std::make_tuple(int64_t(1024), int64_t(1), int64_t(256)))
        copy_kernel<1024, 1, 256, 4><<<1, kThreads, 0, stream>>>(p, o);
    else if (key == std::make_tuple(int64_t(512), int64_t(1), int64_t(256)))
        copy_kernel<512, 1, 256, 2><<<1, kThreads, 0, stream>>>(p, o);
    else
        TORCH_CHECK(false, "copy_tile: no instantiation for n=", out.numel(),
                    " src_stride=", src_stride, " threads=", threads);
    return out.to(at::kFloat);
}

torch::Tensor copy_widen(torch::Tensor x) {
    check_f32_cuda(x, "x");
    auto in = x.to(at::kBFloat16);
    TORCH_CHECK(x.numel() == 256,
                "copy_widen is instantiated for 256 elements");
    auto out = torch::empty_like(x);
    copy_widen_kernel<256, 4>
        <<<1, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const __nv_bfloat16 *>(in.data_ptr()),
            out.data_ptr<float>());
    return out;
}

torch::Tensor tma_stage(torch::Tensor x, int64_t tile, int64_t stages,
                        int64_t src_stride) {
    check_f32_cuda(x, "x");
    const int ntile = int(x.numel() / (tile * src_stride));
    TORCH_CHECK(int64_t(ntile) * tile * src_stride == x.numel(),
                "tile * src_stride must divide x");
    auto out = torch::empty({int64_t(ntile) * tile}, x.options());
    auto stream = at::cuda::getCurrentCUDAStream();
    auto *in = x.data_ptr<float>();
    auto *o = out.data_ptr<float>();
    const auto key = std::make_tuple(tile, stages, src_stride);
    if (key == std::make_tuple(int64_t(1024), int64_t(1), int64_t(1)))
        tma_stage_kernel<1024, 1, 1><<<1, kThreads, 0, stream>>>(in, o, ntile);
    else if (key == std::make_tuple(int64_t(1024), int64_t(3), int64_t(1)))
        tma_stage_kernel<1024, 3, 1><<<1, kThreads, 0, stream>>>(in, o, ntile);
    else if (key == std::make_tuple(int64_t(1024), int64_t(3), int64_t(2)))
        tma_stage_kernel<1024, 3, 2><<<1, kThreads, 0, stream>>>(in, o, ntile);
    else if (key == std::make_tuple(int64_t(6), int64_t(3), int64_t(1)))
        tma_stage_kernel<6, 3, 1><<<1, kThreads, 0, stream>>>(in, o, ntile);
    else
        TORCH_CHECK(false, "tma_stage: no instantiation for tile=", tile,
                    " stages=", stages, " src_stride=", src_stride);
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
    m.def("copy_tile", &copy_tile);
    m.def("copy_widen", &copy_widen);
    m.def("mma_tile", &mma_tile);
}
