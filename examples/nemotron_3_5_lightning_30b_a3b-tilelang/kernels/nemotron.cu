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

constexpr int E = 128;
constexpr int KTOP = 6;
constexpr int I = 1856;
constexpr int IS = 3712;
constexpr float RSCALE = 2.5f;

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

/// Which of the two gemv shapes wins is a measurement, and both are here so it
/// can be repeated. Measured on one H200, one layer of the checkpoint:
///
///   Mamba-2 layer   staged 49.6 us   direct 85.7 us
///   MoE layer       staged  320 us   direct  114 us (with the router fixed)
///
/// The split is how many tiles a CTA gets. The Mamba projections give each CTA
/// about 78 rows, and running three tiles ahead hides the next load behind this
/// tile's fold. The MoE experts give each CTA under two tiles, so the prologue
/// issues more than the loop consumes and the staging is pure overhead. Staged
/// for the first, direct for the second.

/// ``out[row] = dot(w[row], x)`` straight from global memory, no staging.
///
/// A gemv reads each weight byte exactly once, so there is nothing for a shared
/// tile to be reused by. What staging costs instead is occupancy: three stages
/// of eight 2688-wide rows is 129 KB, which fits one CTA per SM and leaves the
/// memory system with one CTA's worth of requests in flight. This reads 16
/// bytes per lane straight from gmem -- a warp covers 512 bytes a step, fully
/// coalesced -- and keeps the shared budget at zero, so the SM holds as many
/// CTAs as the launch offers.
///
/// Kept beside `gemv_rows` rather than replacing it: which one wins is a
/// measurement, and the two are here so the measurement can be repeated.
template <int IN>
__device__ void gemv_rows_direct(const bf16 *__restrict__ w,
                                 const bf16 *__restrict__ x,
                                 bf16 *__restrict__ out, int row0, int nrows,
                                 const bf16 *__restrict__ resid = nullptr) {
    static_assert(IN % 8 == 0, "gemv_rows_direct: IN must be a multiple of 8");
    constexpr int kVec = IN / 8;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int4 *xv = reinterpret_cast<const int4 *>(x);

    for (int r = warp; r < nrows; r += WARPS) {
        const int row = row0 + r;
        const int4 *wv = reinterpret_cast<const int4 *>(w + size_t(row) * IN);
        // Eight accumulators, not one. A single running sum makes the whole row
        // a serial chain of dependent FMAs, and the loads cannot run ahead of a
        // chain: the row's latency, not its bytes, is what the warp waits on.
        float part[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (int v = lane; v < kVec; v += 32) {
            const int4 a = wv[v], b = xv[v];
            const bf16 *pa = reinterpret_cast<const bf16 *>(&a);
            const bf16 *pb = reinterpret_cast<const bf16 *>(&b);
#pragma unroll
            for (int k = 0; k < 8; ++k) part[k] += ld(pa + k) * ld(pb + k);
        }
        float acc = (part[0] + part[1]) + (part[2] + part[3]) +
                    ((part[4] + part[5]) + (part[6] + part[7]));
        acc = ops::warp_reduce<ops::warp_sum>(acc);
        if (lane == 0)
            out[row] = __float2bfloat16(resid ? ld(resid + row) + bf(acc) : acc);
    }
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

constexpr int UP_TILES = I / TILE_ROWS;
constexpr int DOWN_TILES = H / TILE_ROWS;
constexpr int SH_UP_TILES = IS / TILE_ROWS;
constexpr int MOE_STAGES = 3;

/// The router's logits: 128 numbers, spread over the whole grid.
///
/// One warp per expert row, lanes striding it 16 bytes at a time. Written out
/// because the obvious shape -- one *thread* per expert, walking its row -- puts
/// consecutive threads 5376 bytes apart and coalesces nothing: measured at
/// 236 us for 688 KB, against 74 us for the 160 MB of expert weights beside it.
///
/// Every CTA normalises the hidden row itself, as elsewhere; CTA zero also
/// publishes it, because the expert projections read it next.
__global__ __launch_bounds__(THREADS) void k_moe_logits(
    const bf16 *__restrict__ h, const bf16 *__restrict__ gamma,
    const bf16 *__restrict__ w_router, bf16 *__restrict__ h2,
    float *__restrict__ logits) {
    __shared__ float slots[WARPS];
    __shared__ __align__(16) bf16 xs[H];

    rms_norm_to_smem(h, gamma, xs, slots);
    ops::sync<ops::SyncKind::syncthreads>();
    if (blockIdx.x == 0)
        for (int i = int(threadIdx.x); i < H; i += THREADS) h2[i] = xs[i];

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    constexpr int kVec = H / 8;
    const int4 *xv = reinterpret_cast<const int4 *>(xs);
    const int warps_total = int(gridDim.x) * WARPS;
    const int my_warp = int(blockIdx.x) * WARPS + warp;

    for (int e = my_warp; e < E; e += warps_total) {
        const int4 *wv = reinterpret_cast<const int4 *>(w_router + size_t(e) * H);
        float part[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (int v = lane; v < kVec; v += 32) {
            const int4 a = wv[v], b = xv[v];
            const bf16 *pa = reinterpret_cast<const bf16 *>(&a);
            const bf16 *pb = reinterpret_cast<const bf16 *>(&b);
#pragma unroll
            for (int k = 0; k < 8; ++k) part[k] += ld(pa + k) * ld(pb + k);
        }
        float acc = (part[0] + part[1]) + (part[2] + part[3]) +
                    ((part[4] + part[5]) + (part[6] + part[7]));
        acc = ops::warp_reduce<ops::warp_sum>(acc);
        // The router runs in f32 on both sides -- `h2.float() @ w.float().t()`
        // -- so this is the one projection that does not land in bf16.
        if (lane == 0) logits[e] = acc;
    }
}

/// Which six experts, and with what weights.
///
/// 128 logits and six passes over them: one CTA, and the choice is not divided.
/// Dividing it would cost a barrier to put back together for less work than the
/// barrier.
__global__ __launch_bounds__(32) void k_moe_topk(
    const float *__restrict__ logits, const float *__restrict__ e_bias,
    int *__restrict__ idx, float *__restrict__ gw) {
    __shared__ float sig[E];
    __shared__ float ch[E];
    for (int e = int(threadIdx.x); e < E; e += 32) {
        const float s = 1.0f / (1.0f + expf(-logits[e]));
        sig[e] = s;
        ch[e] = s + e_bias[e];
    }
    ops::sync<ops::SyncKind::syncthreads>();
    if (threadIdx.x != 0) return;
    float total = 0.0f;
    for (int j = 0; j < KTOP; ++j) {
        int best = 0;
        for (int e = 1; e < E; ++e)
            if (ch[e] > ch[best]) best = e;
        idx[j] = best;
        total += sig[best];
        ch[best] = -INFINITY;
    }
    const float inv = 1.0f / (total + 1e-20f);
    for (int j = 0; j < KTOP; ++j) gw[j] = sig[idx[j]] * inv * RSCALE;
}

/// One expert's up projection: ``square(relu(h2 @ w_up[e].t()))``.
///
/// ``slot`` selects which of the chosen experts this launch runs, so the six
/// are six launches here and six stages of one kernel once the grid is
/// persistent.
__global__ __launch_bounds__(THREADS) void k_moe_up(
    const bf16 *__restrict__ h2, const bf16 *__restrict__ w_up,
    const int *__restrict__ idx, bf16 *__restrict__ mid) {
    /// `blockIdx.y` is which of the chosen experts this CTA runs. The six are
    /// independent, so they are one launch: at batch 1 each projection moves
    /// about 10 MB, which is microseconds of bandwidth against a launch apiece.
    const int slot = int(blockIdx.y);
    __shared__ __align__(16) bf16 xs[H];
    for (int i = int(threadIdx.x); i < H; i += THREADS) xs[i] = h2[i];
    ops::sync<ops::SyncKind::syncthreads>();
    int first, count;
    tile_span(UP_TILES, int(blockIdx.x), int(gridDim.x), &first, &count);
    const bf16 *w = w_up + size_t(idx[slot]) * I * H;
    bf16 *out = mid + size_t(slot) * I;
    const int lo = first * TILE_ROWS, hi = (first + count) * TILE_ROWS;
    gemv_rows_direct<H>(w, xs, out, lo, hi - lo);
    ops::sync<ops::SyncKind::syncthreads>();
    for (int row = lo + int(threadIdx.x); row < hi; row += THREADS) {
        const float v = fmaxf(ld(out + row), 0.0f);
        out[row] = __float2bfloat16(bf(v) * bf(v));
    }
}

/// One expert's down projection, scaled by its routing weight and accumulated.
///
/// The accumulator is f32 because the reference sums the experts in f32 and
/// only lands in bf16 once, after the six.
__global__ __launch_bounds__(THREADS) void k_moe_down(
    const bf16 *__restrict__ mid, const bf16 *__restrict__ w_down,
    const int *__restrict__ idx, const float *__restrict__ gw,
    float *__restrict__ acc) {
    const int slot = int(blockIdx.y);
    __shared__ __align__(16) bf16 xs[I];
    const bf16 *my_mid = mid + size_t(slot) * I;
    for (int i = int(threadIdx.x); i < I; i += THREADS) xs[i] = my_mid[i];
    ops::sync<ops::SyncKind::syncthreads>();

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    int first, count;
    tile_span(DOWN_TILES, int(blockIdx.x), int(gridDim.x), &first, &count);
    const bf16 *w = w_down + size_t(idx[slot]) * H * I;
    const float scale = gw[slot];
    const int lo = first * TILE_ROWS, hi = (first + count) * TILE_ROWS;
    constexpr int kVec = I / 8;
    const int4 *xv = reinterpret_cast<const int4 *>(xs);

    for (int row = lo + warp; row < hi; row += WARPS) {
        const int4 *wv = reinterpret_cast<const int4 *>(w + size_t(row) * I);
        float part[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (int v = lane; v < kVec; v += 32) {
            const int4 a = wv[v], b = xv[v];
            const bf16 *pa = reinterpret_cast<const bf16 *>(&a);
            const bf16 *pb = reinterpret_cast<const bf16 *>(&b);
#pragma unroll
            for (int k = 0; k < 8; ++k) part[k] += ld(pa + k) * ld(pb + k);
        }
        float dot = (part[0] + part[1]) + (part[2] + part[3]) +
                    ((part[4] + part[5]) + (part[6] + part[7]));
        dot = ops::warp_reduce<ops::warp_sum>(dot);
        // Six experts land on the same row, so the accumulation is atomic. The
        // reference sums them in f32 too; only the order differs, which is a
        // part in 1e7 against a budget of parts in 1e3.
        if (lane == 0) atomicAdd(&acc[row], bf(dot) * scale);
    }
}

/// The shared expert's up projection. Every token goes through it, so it is not
/// selected and takes no routing weight.
__global__ __launch_bounds__(THREADS) void k_moe_shared_up(
    const bf16 *__restrict__ h2, const bf16 *__restrict__ w_sh_up,
    bf16 *__restrict__ smid) {
    __shared__ __align__(16) bf16 xs[H];
    for (int i = int(threadIdx.x); i < H; i += THREADS) xs[i] = h2[i];
    ops::sync<ops::SyncKind::syncthreads>();
    int first, count;
    tile_span(SH_UP_TILES, int(blockIdx.x), int(gridDim.x), &first, &count);
    const int lo = first * TILE_ROWS, hi = (first + count) * TILE_ROWS;
    gemv_rows_direct<H>(w_sh_up, xs, smid, lo, hi - lo);
    ops::sync<ops::SyncKind::syncthreads>();
    for (int row = lo + int(threadIdx.x); row < hi; row += THREADS) {
        const float v = fmaxf(ld(smid + row), 0.0f);
        smid[row] = __float2bfloat16(bf(v) * bf(v));
    }
}

/// The shared expert's down projection, the routed sum, and the residual.
///
/// The routed experts land in bf16 once, after all six have been summed in f32
/// -- ``(total.to(bf16) + sh)`` -- so the accumulator arrives here still in f32.
__global__ __launch_bounds__(THREADS) void k_moe_finish(
    const bf16 *__restrict__ smid, const bf16 *__restrict__ w_sh_down,
    const float *__restrict__ acc, const bf16 *__restrict__ h_in,
    bf16 *__restrict__ h_out) {
    __shared__ __align__(16) bf16 xs[IS];
    for (int i = int(threadIdx.x); i < IS; i += THREADS) xs[i] = smid[i];
    ops::sync<ops::SyncKind::syncthreads>();

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    int first, count;
    tile_span(DOWN_TILES, int(blockIdx.x), int(gridDim.x), &first, &count);
    const int lo = first * TILE_ROWS, hi = (first + count) * TILE_ROWS;
    constexpr int kVec = IS / 8;
    const int4 *xv = reinterpret_cast<const int4 *>(xs);

    for (int row = lo + warp; row < hi; row += WARPS) {
        const int4 *wv = reinterpret_cast<const int4 *>(w_sh_down + size_t(row) * IS);
        float part[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (int v = lane; v < kVec; v += 32) {
            const int4 a = wv[v], b = xv[v];
            const bf16 *pa = reinterpret_cast<const bf16 *>(&a);
            const bf16 *pb = reinterpret_cast<const bf16 *>(&b);
#pragma unroll
            for (int k = 0; k < 8; ++k) part[k] += ld(pa + k) * ld(pb + k);
        }
        float dot = (part[0] + part[1]) + (part[2] + part[3]) +
                    ((part[4] + part[5]) + (part[6] + part[7]));
        dot = ops::warp_reduce<ops::warp_sum>(dot);
        if (lane == 0) {
            // The routed experts land in bf16 once, after all six have summed
            // in f32 -- `(total.to(bf16) + sh)`.
            const float mix = bf(bf(acc[row]) + bf(dot));
            h_out[row] = __float2bfloat16(ld(h_in + row) + mix);
        }
    }
}

/// One MoE layer: router, six routed experts, the shared expert, the residual.
std::vector<torch::Tensor> moe_layer(
    torch::Tensor h, torch::Tensor gamma, torch::Tensor w_router,
    torch::Tensor e_bias, torch::Tensor w_up, torch::Tensor w_down,
    torch::Tensor w_sh_up, torch::Tensor w_sh_down) {
    check_bf16(h, H, "h");
    check_bf16(gamma, H, "gamma");
    check_bf16(w_router, int64_t(E) * H, "w_router");
    TORCH_CHECK(e_bias.is_cuda() && e_bias.is_contiguous() &&
                    e_bias.scalar_type() == at::kFloat && e_bias.numel() == E,
                "e_bias must be a contiguous f32 CUDA tensor of E");
    check_bf16(w_up, int64_t(E) * I * H, "w_up");
    check_bf16(w_down, int64_t(E) * H * I, "w_down");
    check_bf16(w_sh_up, int64_t(IS) * H, "w_sh_up");
    check_bf16(w_sh_down, int64_t(H) * IS, "w_sh_down");

    const auto bf_opt = h.options();
    const auto f_opt = h.options().dtype(at::kFloat);
    auto h2 = torch::empty({H}, bf_opt);
    auto idx = torch::empty({KTOP}, h.options().dtype(at::kInt));
    auto gw = torch::empty({KTOP}, f_opt);
    auto mid = torch::empty({KTOP, I}, bf_opt);
    auto smid = torch::empty({IS}, bf_opt);
    auto acc = torch::zeros({H}, f_opt);
    auto h_out = torch::empty({H}, bf_opt);

    auto stream = at::cuda::getCurrentCUDAStream();
    const int ctas = sm_count();
    // No dynamic shared memory: these read their weights straight from gmem,
    // and a dynamic allocation would cap how many CTAs an SM can hold for
    // nothing in return.

    auto *H2 = static_cast<bf16 *>(h2.data_ptr());
    auto *IDX = static_cast<int *>(idx.data_ptr());
    auto *GW = static_cast<float *>(gw.data_ptr());
    auto *MID = static_cast<bf16 *>(mid.data_ptr());

    auto logits = torch::empty({E}, f_opt);
    k_moe_logits<<<ctas, THREADS, 0, stream>>>(
        static_cast<const bf16 *>(h.data_ptr()),
        static_cast<const bf16 *>(gamma.data_ptr()),
        static_cast<const bf16 *>(w_router.data_ptr()), H2,
        static_cast<float *>(logits.data_ptr()));
    k_moe_topk<<<1, 32, 0, stream>>>(
        static_cast<const float *>(logits.data_ptr()),
        static_cast<const float *>(e_bias.data_ptr()), IDX, GW);
    k_moe_up<<<dim3(ctas, KTOP), THREADS, 0, stream>>>(
        H2, static_cast<const bf16 *>(w_up.data_ptr()), IDX, MID);
    k_moe_down<<<dim3(ctas, KTOP), THREADS, 0, stream>>>(
        MID, static_cast<const bf16 *>(w_down.data_ptr()), IDX, GW,
        static_cast<float *>(acc.data_ptr()));
    k_moe_shared_up<<<ctas, THREADS, 0, stream>>>(
        H2, static_cast<const bf16 *>(w_sh_up.data_ptr()),
        static_cast<bf16 *>(smid.data_ptr()));
    k_moe_finish<<<ctas, THREADS, 0, stream>>>(
        static_cast<const bf16 *>(smid.data_ptr()),
        static_cast<const bf16 *>(w_sh_down.data_ptr()),
        static_cast<const float *>(acc.data_ptr()),
        static_cast<const bf16 *>(h.data_ptr()),
        static_cast<bf16 *>(h_out.data_ptr()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {h_out, idx, gw, h2, mid, smid, acc};
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mamba_layer", &mamba_layer);
    m.def("moe_layer", &moe_layer);
}
