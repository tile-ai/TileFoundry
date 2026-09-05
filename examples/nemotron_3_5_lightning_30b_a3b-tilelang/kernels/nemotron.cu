/// Nemotron-3.5-Lightning-30B-A3B decode, handwritten.
///
/// Every stage is a `__device__` function over gmem pointers plus a shared
/// scratch. The `__global__` wrappers below launch one stage at a time, which
/// is what makes a stage checkable on its own against the op-by-op path; a
/// persistent cooperative grid calls the same functions with a grid barrier
/// where a wrapper boundary is today.

/// Primitives come from the runtime, not from this file:
/// `tilefoundry::ops::{warp_reduce, shuffle_elect, mbarrier_*, tma_bulk_copy}`.
/// The weight staging below is the shape the tilelang implementation uses --
/// one elected lane issues a bulk copy of eight weight rows and arrives on that
/// tile's mbarrier, consumers wait on the phase parity -- expressed through
/// those entries instead of through inline PTX.
#include <tilefoundry/runtime/cuda/runtime.cuh>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

#include <set>
#include <vector>

namespace ops = tilefoundry::ops;

namespace {

constexpr int H = 2688;
constexpr int MH = 64;
constexpr int MD = 64;
constexpr int SS = 128;
constexpr int NG = 8;
constexpr int MI = 4096;
constexpr int CONV = 6144;
constexpr int KER = 4;
constexpr int WIN = 3;
constexpr int PROJ = 10304;
constexpr int GRP = 512;
constexpr int HPG = 8;
constexpr float EPS = 1e-5f;
constexpr float DTMIN = 0.001f;

constexpr int THREADS = 256;
constexpr int WARPS = THREADS / 32;

using bf16 = __nv_bfloat16;

/// Round to bf16 and carry on in f32, which is what a bf16 op does.
__device__ inline float bf(float x) {
    return __bfloat162float(__float2bfloat16(x));
}

__device__ inline float ld(const bf16 *p) { return __bfloat162float(*p); }

__device__ inline float silu_f(float x) { return x / (1.0f + expf(-x)); }

/// ``F.softplus``: log1p(exp(x)), with the large-x branch the library takes so
/// a big activation does not overflow the exponential.
__device__ inline float softplus_f(float x) {
    return x > 20.0f ? x : log1pf(expf(x));
}

/// Sum ``v`` across the whole CTA, leaving the total in every thread.
///
/// A warp fold through ``ops::warp_reduce`` and then one slot per warp in
/// shared memory; the second fold is over eight values, so it costs less as a
/// serial loop than as a second butterfly.
__device__ inline float cta_sum(float v, float *slots) {
    v = ops::warp_reduce<ops::warp_sum>(v);
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) slots[warp] = v;
    ops::sync<ops::SyncKind::syncthreads>();
    float total = 0.0f;
    for (int i = 0; i < WARPS; ++i) total += slots[i];
    return total;
}


/// One weight tile: ``TILE_ROWS`` whole rows of an ``(out, in)`` matrix.
///
/// A CTA owns a run of output rows, so its slice is a contiguous byte run and
/// the rank-1 bulk copy is the instruction that moves it. Eight rows is what
/// makes the run a whole number of 16-byte grains for both widths this model
/// uses, and it is one row per warp.
constexpr int TILE_ROWS = WARPS;

/// ``out[row] = dot(w[row], x)`` for a run of rows, streaming ``w`` through a
/// shared ring.
///
/// The loads run ``STAGES`` tiles ahead, so the tile issued last is still in
/// flight while this one is being folded. Every fold is a warp butterfly, and
/// the accumulator is f32 with a bf16 landing on the way out -- what a bf16
/// matmul does.
template <int IN, int STAGES>
__device__ void gemv_rows(const bf16 *__restrict__ w, const bf16 *__restrict__ x,
                          bf16 *__restrict__ out, int tile0, int ntiles,
                          bf16 *stage, uint64_t *bars,
                          const bf16 *__restrict__ resid = nullptr) {
    constexpr unsigned kBytes = unsigned(TILE_ROWS) * IN * sizeof(bf16);
    constexpr int kTileElems = TILE_ROWS * IN;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;

    const bool elected = ops::shuffle_elect<THREADS>();
    if (elected)
        for (int s = 0; s < STAGES; ++s) ops::mbarrier_init(&bars[s], 1);
    ops::sync<ops::SyncKind::syncthreads>();

    for (int s = 0; s < STAGES && s < ntiles; ++s) {
        if (elected) {
            ops::mbarrier_arrive_expect_tx(&bars[s], kBytes);
            ops::tma_bulk_copy(w + size_t(tile0 + s) * kTileElems,
                               stage + s * kTileElems, kTileElems, &bars[s]);
        }
    }

    for (int t = 0; t < ntiles; ++t) {
        const int slot = t % STAGES;
        ops::mbarrier_wait_parity(&bars[slot], unsigned(t / STAGES) & 1u);

        const bf16 *row = stage + slot * kTileElems + warp * IN;
        float acc = 0.0f;
        for (int j = lane; j < IN; j += 32) acc += ld(row + j) * ld(x + j);
        acc = ops::warp_reduce<ops::warp_sum>(acc);
        if (lane == 0) {
            const int row = (tile0 + t) * TILE_ROWS + warp;
            // The product lands in bf16 before the residual adds to it, which
            // is what `h + (scan @ w_out.t())` does one operation at a time.
            out[row] = __float2bfloat16(resid ? ld(resid + row) + bf(acc) : acc);
        }

        ops::sync<ops::SyncKind::syncthreads>();
        const int next = t + STAGES;
        if (next < ntiles && elected) {
            ops::mbarrier_arrive_expect_tx(&bars[slot], kBytes);
            ops::tma_bulk_copy(w + size_t(tile0 + next) * kTileElems,
                               stage + slot * kTileElems, kTileElems, &bars[slot]);
        }
    }
    if (elected)
        for (int s = 0; s < STAGES; ++s) ops::mbarrier_invalidate(&bars[s]);
}

/// The published ``NemotronHRMSNorm``, into shared memory.
///
/// The normalised value lands in bf16 **before** the weight multiplies it,
/// which is one rounding more than doing the whole thing in f32 would be. It is
/// what the checkpoint's own implementation does, so it is what this does.
__device__ void rms_norm_to_smem(const bf16 *__restrict__ h,
                                 const bf16 *__restrict__ gamma,
                                 bf16 *__restrict__ dst, float *slots) {
    float sq = 0.0f;
    for (int i = threadIdx.x; i < H; i += THREADS) {
        const float v = ld(h + i);
        sq += v * v;
    }
    const float scale = rsqrtf(cta_sum(sq, slots) / float(H) + EPS);
    for (int i = threadIdx.x; i < H; i += THREADS)
        dst[i] = __float2bfloat16(bf(ld(h + i) * scale) * ld(gamma + i));
}

/// The depthwise convolution over one channel run, and the state it hands on.
///
/// The window a step hands on is two columns of the window it was handed,
/// shifted, plus one fresh column from the input projection. Every rounding
/// below is one the op-by-op reference makes: the products land in bf16, the
/// fold over the four taps is f32, and the bias and the silu each land again.
__device__ void mamba_conv_stage(const bf16 *__restrict__ col0,
                                 const bf16 *__restrict__ conv_state,
                                 const bf16 *__restrict__ conv_w,
                                 const bf16 *__restrict__ conv_b,
                                 bf16 *__restrict__ conv_out,
                                 bf16 *__restrict__ xbc, int c0, int c1) {
    for (int c = c0 + int(threadIdx.x); c < c1; c += THREADS) {
        float win[KER];
        for (int k = 0; k < WIN; ++k) win[k] = ld(conv_state + c * WIN + k);
        win[WIN] = ld(col0 + c);
        for (int k = 0; k < WIN; ++k)
            conv_out[c * WIN + k] = __float2bfloat16(win[k + 1]);
        float acc = 0.0f;
        for (int k = 0; k < KER; ++k) acc += bf(win[k] * ld(conv_w + c * KER + k));
        xbc[c] = __float2bfloat16(silu_f(bf(bf(acc) + ld(conv_b + c))));
    }
}

/// One head's recurrence: advance the state and contract it against C.
///
/// ``head`` is passed rather than read from ``blockIdx`` so a persistent grid
/// can call this for whichever head the CTA owns. Warp ``w`` takes the ``d``
/// rows ``w, w + 8, ...`` and its lanes take ``s`` in strides of 32, so the
/// contraction over ``s`` is a warp fold and nothing crosses a warp.
__device__ void mamba_ssm_stage(const bf16 *__restrict__ xbc,
                                const bf16 *__restrict__ dt,
                                const bf16 *__restrict__ mscal,
                                const float *__restrict__ ssm_in,
                                float *__restrict__ ssm_out,
                                bf16 *__restrict__ y, int head) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int group = head / HPG;

    const float a_log = ld(mscal + 0 * MH + head);
    const float dt_bias = ld(mscal + 1 * MH + head);
    const float dta = bf(fmaxf(bf(softplus_f(bf(ld(dt + head) + dt_bias))), DTMIN));
    const float da = expf(dta * (-expf(a_log)));

    const bf16 *bg = xbc + MI + group * SS;
    const bf16 *cg = xbc + MI + NG * SS + group * SS;

    for (int d = warp; d < MD; d += WARPS) {
        const float xv = ld(xbc + head * MD + d);
        float acc = 0.0f;
        for (int s = lane; s < SS; s += 32) {
            const size_t at = (size_t(head) * MD + d) * SS + s;
            const float dbx = bf(bf(dta * ld(bg + s)) * xv);
            const float so = ssm_in[at] * da + dbx;
            ssm_out[at] = so;
            acc += bf(so) * ld(cg + s);
        }
        acc = ops::warp_reduce<ops::warp_sum>(acc);
        if (lane == 0) y[head * MD + d] = __float2bfloat16(acc);
    }
}

/// The D skip, the gate, and the gated group norm over one group.
///
/// ``group`` names which of the ``NG`` runs of ``GRP`` this call owns. The norm
/// is over that run alone, so the CTA-wide fold below is the whole reduction.
__device__ void mamba_gate_norm_stage(const bf16 *__restrict__ y,
                                      const bf16 *__restrict__ xbc,
                                      const bf16 *__restrict__ gate,
                                      const bf16 *__restrict__ mscal,
                                      const bf16 *__restrict__ ggdn,
                                      bf16 *__restrict__ scan, int group,
                                      float *slots, float *yf) {
    const int base = group * GRP;
    float sq = 0.0f;
    for (int j = int(threadIdx.x); j < GRP; j += THREADS) {
        const int i = base + j;
        const int head = i / MD;
        const float xv = ld(xbc + i);
        const float d_skip = ld(mscal + 2 * MH + head);
        const float yd = bf(ld(y + i) + bf(xv * d_skip));
        const float v = yd * silu_f(ld(gate + i));
        yf[j] = v;
        sq += v * v;
    }
    const float scale = rsqrtf(cta_sum(sq, slots) / float(GRP) + EPS);
    for (int j = int(threadIdx.x); j < GRP; j += THREADS)
        scan[base + j] = __float2bfloat16(bf(yf[j] * scale) * ld(ggdn + base + j));
}

/// The tile range CTA ``c`` of ``nc`` owns, splitting ``ntiles`` as evenly as
/// an integer division allows.
__device__ __host__ inline void tile_span(int ntiles, int c, int nc, int *first,
                                          int *count) {
    const int lo = int(int64_t(ntiles) * c / nc);
    const int hi = int(int64_t(ntiles) * (c + 1) / nc);
    *first = lo;
    *count = hi - lo;
}

/// Shared layout shared by both projections: the staged weight ring first (the
/// bulk copy's destination has the strictest alignment), then the input vector,
/// then the barriers and the per-warp fold slots.
template <int IN, int STAGES> struct ProjSmem {
    bf16 *stage;
    bf16 *x;
    uint64_t *bars;
    float *slots;

    __device__ explicit ProjSmem(char *raw) {
        stage = reinterpret_cast<bf16 *>(raw);
        x = stage + size_t(STAGES) * TILE_ROWS * IN;
        bars = reinterpret_cast<uint64_t *>(x + IN);
        slots = reinterpret_cast<float *>(bars + STAGES);
    }
    static constexpr size_t bytes() {
        return sizeof(bf16) * (size_t(STAGES) * TILE_ROWS * IN + IN) +
               sizeof(uint64_t) * STAGES + sizeof(float) * WARPS;
    }
};

constexpr int IN_STAGES = 3;
constexpr int OUT_STAGES = 2;
constexpr int PROJ_TILES = PROJ / TILE_ROWS;
constexpr int OUT_TILES = H / TILE_ROWS;

/// Pre-norm and the input projection, in one launch.
///
/// Every CTA normalises the hidden row itself rather than one CTA doing it
/// behind a barrier: 2688 elements is not work worth dividing, and dividing it
/// would cost a barrier to put back together.
__global__ __launch_bounds__(THREADS) void k_in_proj(
    const bf16 *__restrict__ h, const bf16 *__restrict__ gamma,
    const bf16 *__restrict__ w_in, bf16 *__restrict__ proj) {
    extern __shared__ __align__(16) char raw[];
    ProjSmem<H, IN_STAGES> sm(raw);
    rms_norm_to_smem(h, gamma, sm.x, sm.slots);
    ops::sync<ops::SyncKind::syncthreads>();
    int first, count;
    tile_span(PROJ_TILES, int(blockIdx.x), int(gridDim.x), &first, &count);
    gemv_rows<H, IN_STAGES>(w_in, sm.x, proj, first, count, sm.stage, sm.bars);
}

/// The output projection and the residual add.
__global__ __launch_bounds__(THREADS) void k_out_proj(
    const bf16 *__restrict__ scan, const bf16 *__restrict__ w_out,
    const bf16 *__restrict__ h_in, bf16 *__restrict__ h_out) {
    extern __shared__ __align__(16) char raw[];
    ProjSmem<MI, OUT_STAGES> sm(raw);
    for (int i = int(threadIdx.x); i < MI; i += THREADS) sm.x[i] = scan[i];
    ops::sync<ops::SyncKind::syncthreads>();
    int first, count;
    tile_span(OUT_TILES, int(blockIdx.x), int(gridDim.x), &first, &count);
    gemv_rows<MI, OUT_STAGES>(w_out, sm.x, h_out, first, count, sm.stage, sm.bars,
                              h_in);
}

__global__ __launch_bounds__(THREADS) void k_conv(
    const bf16 *__restrict__ proj, const bf16 *__restrict__ conv_state,
    const bf16 *__restrict__ conv_w, const bf16 *__restrict__ conv_b,
    bf16 *__restrict__ conv_out, bf16 *__restrict__ xbc) {
    const int per = (CONV + int(gridDim.x) - 1) / int(gridDim.x);
    const int c0 = int(blockIdx.x) * per;
    mamba_conv_stage(proj + MI, conv_state, conv_w, conv_b, conv_out, xbc, c0,
                     min(c0 + per, CONV));
}

__global__ __launch_bounds__(THREADS) void k_ssm(
    const bf16 *__restrict__ xbc, const bf16 *__restrict__ proj,
    const bf16 *__restrict__ mscal, const float *__restrict__ ssm_in,
    float *__restrict__ ssm_out, bf16 *__restrict__ y) {
    mamba_ssm_stage(xbc, proj + MI + CONV, mscal, ssm_in, ssm_out, y,
                    int(blockIdx.x));
}

__global__ __launch_bounds__(THREADS) void k_gate_norm(
    const bf16 *__restrict__ y, const bf16 *__restrict__ xbc,
    const bf16 *__restrict__ proj, const bf16 *__restrict__ mscal,
    const bf16 *__restrict__ ggdn, bf16 *__restrict__ scan) {
    __shared__ float slots[WARPS];
    __shared__ float yf[GRP];
    mamba_gate_norm_stage(y, xbc, proj, mscal, ggdn, scan, int(blockIdx.x), slots,
                          yf);
}

/// Number of resident CTAs the projections launch with: one per SM, queried
/// rather than assumed so the split does not depend on which card this is.
int sm_count() {
    static int n = 0;
    if (n == 0) n = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    return n;
}

/// Raise a kernel's dynamic shared-memory ceiling past the 48 KB default.
template <class Fn> void allow_smem(Fn *fn, size_t bytes) {
    static thread_local std::set<void *> done;
    if (done.insert(reinterpret_cast<void *>(fn)).second)
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            reinterpret_cast<const void *>(fn),
            cudaFuncAttributeMaxDynamicSharedMemorySize, int(bytes)));
}

void check_bf16(const torch::Tensor &t, int64_t n, const char *what) {
    TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.scalar_type() == at::kBFloat16,
                what, " must be a contiguous bf16 CUDA tensor");
    TORCH_CHECK(t.numel() == n, what, " must hold ", n, " elements, got ", t.numel());
}

/// One Mamba-2 layer, stage by stage.
///
/// Returns every intermediate, not only the three the model carries forward: a
/// stage that disagrees with the op-by-op reference is then located directly
/// rather than bisected out of a single wrong hidden row.
std::vector<torch::Tensor> mamba_layer(
    torch::Tensor h, torch::Tensor gamma, torch::Tensor w_in, torch::Tensor w_out,
    torch::Tensor conv_w, torch::Tensor conv_b, torch::Tensor ggdn,
    torch::Tensor mscal, torch::Tensor conv_state, torch::Tensor ssm_state) {
    check_bf16(h, H, "h");
    check_bf16(gamma, H, "gamma");
    check_bf16(w_in, int64_t(PROJ) * H, "w_in");
    check_bf16(w_out, int64_t(H) * MI, "w_out");
    check_bf16(conv_w, int64_t(CONV) * KER, "conv_w");
    check_bf16(conv_b, CONV, "conv_b");
    check_bf16(ggdn, MI, "ggdn");
    check_bf16(mscal, 3 * MH, "mscal");
    check_bf16(conv_state, int64_t(CONV) * WIN, "conv_state");
    TORCH_CHECK(ssm_state.is_cuda() && ssm_state.is_contiguous() &&
                    ssm_state.scalar_type() == at::kFloat &&
                    ssm_state.numel() == int64_t(MH) * MD * SS,
                "ssm_state must be a contiguous f32 CUDA tensor of MH*MD*SS");

    const auto bf_opt = h.options();
    auto proj = torch::empty({PROJ}, bf_opt);
    auto xbc = torch::empty({CONV}, bf_opt);
    auto y = torch::empty({MI}, bf_opt);
    auto scan = torch::empty({MI}, bf_opt);
    auto h_out = torch::empty({H}, bf_opt);
    auto conv_out = torch::empty({int64_t(CONV) * WIN}, bf_opt);
    auto ssm_out = torch::empty_like(ssm_state);

    auto *P = static_cast<bf16 *>(proj.data_ptr());
    auto *X = static_cast<bf16 *>(xbc.data_ptr());
    auto stream = at::cuda::getCurrentCUDAStream();
    const int ctas = sm_count();

    constexpr size_t kIn = ProjSmem<H, IN_STAGES>::bytes();
    constexpr size_t kOut = ProjSmem<MI, OUT_STAGES>::bytes();
    allow_smem(k_in_proj, kIn);
    allow_smem(k_out_proj, kOut);

    k_in_proj<<<ctas, THREADS, kIn, stream>>>(
        static_cast<const bf16 *>(h.data_ptr()),
        static_cast<const bf16 *>(gamma.data_ptr()),
        static_cast<const bf16 *>(w_in.data_ptr()), P);
    k_conv<<<CONV / THREADS, THREADS, 0, stream>>>(
        P, static_cast<const bf16 *>(conv_state.data_ptr()),
        static_cast<const bf16 *>(conv_w.data_ptr()),
        static_cast<const bf16 *>(conv_b.data_ptr()),
        static_cast<bf16 *>(conv_out.data_ptr()), X);
    k_ssm<<<MH, THREADS, 0, stream>>>(
        X, P, static_cast<const bf16 *>(mscal.data_ptr()),
        static_cast<const float *>(ssm_state.data_ptr()),
        static_cast<float *>(ssm_out.data_ptr()),
        static_cast<bf16 *>(y.data_ptr()));
    k_gate_norm<<<NG, THREADS, 0, stream>>>(
        static_cast<const bf16 *>(y.data_ptr()), X, P,
        static_cast<const bf16 *>(mscal.data_ptr()),
        static_cast<const bf16 *>(ggdn.data_ptr()),
        static_cast<bf16 *>(scan.data_ptr()));
    k_out_proj<<<ctas, THREADS, kOut, stream>>>(
        static_cast<const bf16 *>(scan.data_ptr()),
        static_cast<const bf16 *>(w_out.data_ptr()),
        static_cast<const bf16 *>(h.data_ptr()),
        static_cast<bf16 *>(h_out.data_ptr()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {h_out, conv_out, ssm_out, proj, xbc, y, scan};
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("mamba_layer", &mamba_layer); }
