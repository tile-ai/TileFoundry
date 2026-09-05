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

#include <cooperative_groups.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

#include <set>
#include <vector>

namespace ops = tilefoundry::ops;
namespace cg = cooperative_groups;

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

constexpr int HQ = 32;
constexpr int HKV = 2;
constexpr int DH = 128;
constexpr int QP = 4096;
constexpr int KVP = 256;
constexpr int GQA = 16;
constexpr float QSCALE = 0.08838834764831845f;

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
__device__ void s_in_proj(int cta, int nctas, const bf16 *__restrict__ h, const bf16 *__restrict__ gamma, const bf16 *__restrict__ w_in, bf16 *__restrict__ proj) {
    __shared__ float slots[WARPS];
    __shared__ __align__(16) bf16 xs[H];
    rms_norm_to_smem(h, gamma, xs, slots);
    ops::sync<ops::SyncKind::syncthreads>();
    int first, count;
    tile_span(PROJ_TILES, cta, nctas, &first, &count);
    gemv_rows_direct<H>(w_in, xs, proj, first * TILE_ROWS, count * TILE_ROWS);
}

__global__ __launch_bounds__(THREADS) void k_in_proj(
    const bf16 *__restrict__ h, const bf16 *__restrict__ gamma,
    const bf16 *__restrict__ w_in, bf16 *__restrict__ proj) {
    s_in_proj(int(blockIdx.x), int(gridDim.x), h, gamma, w_in, proj);
}


/// The output projection and the residual add.
__device__ void s_out_proj(int cta, int nctas, const bf16 *__restrict__ scan, const bf16 *__restrict__ w_out, const bf16 *__restrict__ h_in, bf16 *__restrict__ h_out) {
    __shared__ __align__(16) bf16 xs[MI];
    for (int i = int(threadIdx.x); i < MI; i += THREADS) xs[i] = scan[i];
    ops::sync<ops::SyncKind::syncthreads>();
    int first, count;
    tile_span(OUT_TILES, cta, nctas, &first, &count);
    gemv_rows_direct<MI>(w_out, xs, h_out, first * TILE_ROWS, count * TILE_ROWS,
                         h_in);
}

__global__ __launch_bounds__(THREADS) void k_out_proj(
    const bf16 *__restrict__ scan, const bf16 *__restrict__ w_out,
    const bf16 *__restrict__ h_in, bf16 *__restrict__ h_out) {
    s_out_proj(int(blockIdx.x), int(gridDim.x), scan, w_out, h_in, h_out);
}


__device__ void s_conv(int cta, int nctas, const bf16 *__restrict__ proj, const bf16 *__restrict__ conv_state, const bf16 *__restrict__ conv_w, const bf16 *__restrict__ conv_b, bf16 *__restrict__ conv_out, bf16 *__restrict__ xbc) {
    const int per = (CONV + nctas - 1) / nctas;
    const int c0 = cta * per;
    mamba_conv_stage(proj + MI, conv_state, conv_w, conv_b, conv_out, xbc, c0,
                     min(c0 + per, CONV));
}

__global__ __launch_bounds__(THREADS) void k_conv(
    const bf16 *__restrict__ proj, const bf16 *__restrict__ conv_state,
    const bf16 *__restrict__ conv_w, const bf16 *__restrict__ conv_b,
    bf16 *__restrict__ conv_out, bf16 *__restrict__ xbc) {
    s_conv(int(blockIdx.x), int(gridDim.x), proj, conv_state, conv_w, conv_b, conv_out, xbc);
}


__device__ void s_ssm(int cta, int nctas, const bf16 *__restrict__ xbc, const bf16 *__restrict__ proj, const bf16 *__restrict__ mscal, const float *__restrict__ ssm_in, float *__restrict__ ssm_out, bf16 *__restrict__ y) {
    // 64 heads over however many CTAs the grid has. A persistent grid has more
    // of them than there are heads, and a CTA that owns none must not index one.
    for (int head = cta; head < MH; head += nctas)
        mamba_ssm_stage(xbc, proj + MI + CONV, mscal, ssm_in, ssm_out, y, head);
}

__global__ __launch_bounds__(THREADS) void k_ssm(
    const bf16 *__restrict__ xbc, const bf16 *__restrict__ proj,
    const bf16 *__restrict__ mscal, const float *__restrict__ ssm_in,
    float *__restrict__ ssm_out, bf16 *__restrict__ y) {
    s_ssm(int(blockIdx.x), int(gridDim.x), xbc, proj, mscal, ssm_in, ssm_out, y);
}


__device__ void s_gate_norm(int cta, int nctas, const bf16 *__restrict__ y, const bf16 *__restrict__ xbc, const bf16 *__restrict__ proj, const bf16 *__restrict__ mscal, const bf16 *__restrict__ ggdn, bf16 *__restrict__ scan) {
    __shared__ float slots[WARPS];
    __shared__ float yf[GRP];
    // Eight groups, same reasoning as the heads above.
    for (int group = cta; group < NG; group += nctas)
        mamba_gate_norm_stage(y, xbc, proj, mscal, ggdn, scan, group, slots, yf);
}

__global__ __launch_bounds__(THREADS) void k_gate_norm(
    const bf16 *__restrict__ y, const bf16 *__restrict__ xbc,
    const bf16 *__restrict__ proj, const bf16 *__restrict__ mscal,
    const bf16 *__restrict__ ggdn, bf16 *__restrict__ scan) {
    s_gate_norm(int(blockIdx.x), int(gridDim.x), y, xbc, proj, mscal, ggdn, scan);
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
__device__ void s_moe_logits(int cta, int nctas, const bf16 *__restrict__ h, const bf16 *__restrict__ gamma, const bf16 *__restrict__ w_router, bf16 *__restrict__ h2, float *__restrict__ logits) {
    __shared__ float slots[WARPS];
    __shared__ __align__(16) bf16 xs[H];

    rms_norm_to_smem(h, gamma, xs, slots);
    ops::sync<ops::SyncKind::syncthreads>();
    if (cta == 0)
        for (int i = int(threadIdx.x); i < H; i += THREADS) h2[i] = xs[i];

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    constexpr int kVec = H / 8;
    const int4 *xv = reinterpret_cast<const int4 *>(xs);
    const int warps_total = nctas * WARPS;
    const int my_warp = cta * WARPS + warp;

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

__global__ __launch_bounds__(THREADS) void k_moe_logits(
    const bf16 *__restrict__ h, const bf16 *__restrict__ gamma,
    const bf16 *__restrict__ w_router, bf16 *__restrict__ h2,
    float *__restrict__ logits) {
    s_moe_logits(int(blockIdx.x), int(gridDim.x), h, gamma, w_router, h2, logits);
}


/// Which six experts, and with what weights.
///
/// 128 logits and six passes over them: one CTA, and the choice is not divided.
/// Dividing it would cost a barrier to put back together for less work than the
/// barrier.
__device__ void s_moe_topk(int cta, int nctas, const float *__restrict__ logits, const float *__restrict__ e_bias, int *__restrict__ idx, float *__restrict__ gw) {
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

__global__ __launch_bounds__(32) void k_moe_topk(
    const float *__restrict__ logits, const float *__restrict__ e_bias,
    int *__restrict__ idx, float *__restrict__ gw) {
    s_moe_topk(int(blockIdx.x), int(gridDim.x), logits, e_bias, idx, gw);
}


/// One expert's up projection: ``square(relu(h2 @ w_up[e].t()))``.
///
/// ``slot`` selects which of the chosen experts this launch runs, so the six
/// are six launches here and six stages of one kernel once the grid is
/// persistent.
__device__ void s_moe_up(int cta, int nctas, int slot, const bf16 *__restrict__ h2, const bf16 *__restrict__ w_up, const int *__restrict__ idx, bf16 *__restrict__ mid) {
    /// `blockIdx.y` is which of the chosen experts this CTA runs. The six are
    /// independent, so they are one launch: at batch 1 each projection moves
    /// about 10 MB, which is microseconds of bandwidth against a launch apiece.
    __shared__ __align__(16) bf16 xs[H];
    for (int i = int(threadIdx.x); i < H; i += THREADS) xs[i] = h2[i];
    ops::sync<ops::SyncKind::syncthreads>();
    int first, count;
    tile_span(UP_TILES, cta, nctas, &first, &count);
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

__global__ __launch_bounds__(THREADS) void k_moe_up(
    const bf16 *__restrict__ h2, const bf16 *__restrict__ w_up,
    const int *__restrict__ idx, bf16 *__restrict__ mid) {
    s_moe_up(int(blockIdx.x), int(gridDim.x), int(blockIdx.y), h2, w_up, idx, mid);
}


/// One expert's down projection, scaled by its routing weight and accumulated.
///
/// The accumulator is f32 because the reference sums the experts in f32 and
/// only lands in bf16 once, after the six.
__device__ void s_moe_down(int cta, int nctas, int slot, const bf16 *__restrict__ mid, const bf16 *__restrict__ w_down, const int *__restrict__ idx, const float *__restrict__ gw, float *__restrict__ acc) {
    __shared__ __align__(16) bf16 xs[I];
    const bf16 *my_mid = mid + size_t(slot) * I;
    for (int i = int(threadIdx.x); i < I; i += THREADS) xs[i] = my_mid[i];
    ops::sync<ops::SyncKind::syncthreads>();

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    int first, count;
    tile_span(DOWN_TILES, cta, nctas, &first, &count);
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
    // `xs` holds *this* expert's activations. A persistent grid runs the six
    // experts one after another in the same CTA, so without this the next
    // expert overwrites the tile while threads are still folding against it.
    // Six separate launches never had to say so.
    ops::sync<ops::SyncKind::syncthreads>();
}

__global__ __launch_bounds__(THREADS) void k_moe_down(
    const bf16 *__restrict__ mid, const bf16 *__restrict__ w_down,
    const int *__restrict__ idx, const float *__restrict__ gw,
    float *__restrict__ acc) {
    s_moe_down(int(blockIdx.x), int(gridDim.x), int(blockIdx.y), mid, w_down, idx, gw, acc);
}


/// The shared expert's up projection. Every token goes through it, so it is not
/// selected and takes no routing weight.
__device__ void s_moe_shared_up(int cta, int nctas, const bf16 *__restrict__ h2, const bf16 *__restrict__ w_sh_up, bf16 *__restrict__ smid) {
    __shared__ __align__(16) bf16 xs[H];
    for (int i = int(threadIdx.x); i < H; i += THREADS) xs[i] = h2[i];
    ops::sync<ops::SyncKind::syncthreads>();
    int first, count;
    tile_span(SH_UP_TILES, cta, nctas, &first, &count);
    const int lo = first * TILE_ROWS, hi = (first + count) * TILE_ROWS;
    gemv_rows_direct<H>(w_sh_up, xs, smid, lo, hi - lo);
    ops::sync<ops::SyncKind::syncthreads>();
    for (int row = lo + int(threadIdx.x); row < hi; row += THREADS) {
        const float v = fmaxf(ld(smid + row), 0.0f);
        smid[row] = __float2bfloat16(bf(v) * bf(v));
    }
}

__global__ __launch_bounds__(THREADS) void k_moe_shared_up(
    const bf16 *__restrict__ h2, const bf16 *__restrict__ w_sh_up,
    bf16 *__restrict__ smid) {
    s_moe_shared_up(int(blockIdx.x), int(gridDim.x), h2, w_sh_up, smid);
}


/// The shared expert's down projection, the routed sum, and the residual.
///
/// The routed experts land in bf16 once, after all six have been summed in f32
/// -- ``(total.to(bf16) + sh)`` -- so the accumulator arrives here still in f32.
__device__ void s_moe_finish(int cta, int nctas, const bf16 *__restrict__ smid, const bf16 *__restrict__ w_sh_down, const float *__restrict__ acc, const bf16 *__restrict__ h_in, bf16 *__restrict__ h_out) {
    __shared__ __align__(16) bf16 xs[IS];
    for (int i = int(threadIdx.x); i < IS; i += THREADS) xs[i] = smid[i];
    ops::sync<ops::SyncKind::syncthreads>();

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    int first, count;
    tile_span(DOWN_TILES, cta, nctas, &first, &count);
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

__global__ __launch_bounds__(THREADS) void k_moe_finish(
    const bf16 *__restrict__ smid, const bf16 *__restrict__ w_sh_down,
    const float *__restrict__ acc, const bf16 *__restrict__ h_in,
    bf16 *__restrict__ h_out) {
    s_moe_finish(int(blockIdx.x), int(gridDim.x), smid, w_sh_down, acc, h_in, h_out);
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

/// Keys per CTA of the attention scan.
///
/// The scan is split over the context because at 262080 positions one CTA per
/// head would leave 130 of the 132 SMs idle. Each CTA folds its own block into
/// a running (max, sum, accumulator) and `k_attn_combine` merges the blocks,
/// which is what makes reading them separately the same value as one softmax
/// over their concatenation.
constexpr int ABLK = 256;
constexpr int ATT_THREADS = 256;
constexpr int ATT_WARPS = ATT_THREADS / 32;
constexpr int Q_PER_WARP = GQA / ATT_WARPS;

/// One block of keys, for one KV head, folded into a partial softmax.
///
/// Lane `l` takes key `base + l`, so a lane's 128 dot products are serial and no
/// reduction crosses the warp for a score. The softmax over the 32 keys a warp
/// holds is two warp folds, and the value accumulation broadcasts each weight
/// back with a shuffle.
__device__ void s_attn_block(int cta, int nctas, int head, const bf16 *__restrict__ q, const bf16 *__restrict__ kc, const bf16 *__restrict__ vc, int nkey, int blk_offset, float *__restrict__ pm, float *__restrict__ pl, float *__restrict__ pacc) {
    const int blk = cta;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;

    __shared__ __align__(16) bf16 qs[GQA * DH];
    for (int i = int(threadIdx.x); i < GQA * DH; i += ATT_THREADS)
        qs[i] = q[size_t(head) * GQA * DH + i];
    ops::sync<ops::SyncKind::syncthreads>();

    const int lo = blk * ABLK;
    const int hi = min(lo + ABLK, nkey);

    for (int qi = 0; qi < Q_PER_WARP; ++qi) {
        const int g = warp * Q_PER_WARP + qi;
        const bf16 *qrow = qs + g * DH;
        float m = -INFINITY, l = 0.0f;
        float acc[DH / 32];
#pragma unroll
        for (int i = 0; i < DH / 32; ++i) acc[i] = 0.0f;

        for (int base = lo; base < hi; base += 32) {
            const int key = base + lane;
            /// `raw` lands in bf16 before the scale, because
            /// `matmul(qg, kh.t())` is a bf16 matmul and only then `.float()`.
            float s = -INFINITY;
            if (key < hi) {
                const bf16 *krow = kc + (size_t(key) * HKV + head) * DH;
                float part[4] = {0, 0, 0, 0};
#pragma unroll
                for (int d = 0; d < DH; d += 4)
#pragma unroll
                    for (int u = 0; u < 4; ++u)
                        part[u] += ld(krow + d + u) * ld(qrow + d + u);
                s = bf(part[0] + part[1] + part[2] + part[3]) * QSCALE;
            }
            const float bm = ops::warp_reduce<ops::warp_max>(s);
            const float nm = fmaxf(m, bm);
            const float cr = (m == -INFINITY) ? 0.0f : __expf(m - nm);
            const float pw = (key < hi) ? __expf(s - nm) : 0.0f;
            l = l * cr + ops::warp_reduce<ops::warp_sum>(pw);
#pragma unroll
            for (int i = 0; i < DH / 32; ++i) acc[i] *= cr;
            /// The weights are per key, the accumulator per dim, so each key's
            /// weight comes back to every lane with a shuffle rather than
            /// through shared memory.
            for (int j = 0; j < 32; ++j) {
                const float w = __shfl_sync(0xFFFFFFFFu, pw, j);
                const int kj = base + j;
                if (kj >= hi || w == 0.0f) continue;
                const bf16 *vrow = vc + (size_t(kj) * HKV + head) * DH;
#pragma unroll
                for (int i = 0; i < DH / 32; ++i)
                    acc[i] += w * ld(vrow + lane + i * 32);
            }
            m = nm;
        }
        const size_t at = (size_t(blk_offset + blk) * HKV + head) * GQA + g;
        if (lane == 0) {
            pm[at] = m;
            pl[at] = l;
        }
#pragma unroll
        for (int i = 0; i < DH / 32; ++i)
            pacc[at * DH + lane + i * 32] = acc[i];
    }
}

__global__ __launch_bounds__(ATT_THREADS) void k_attn_block(
    const bf16 *__restrict__ q, const bf16 *__restrict__ kc,
    const bf16 *__restrict__ vc, int nkey, int blk_offset,
    float *__restrict__ pm, float *__restrict__ pl, float *__restrict__ pacc) {
    s_attn_block(int(blockIdx.x), int(gridDim.x), int(blockIdx.y), q, kc, vc, nkey, blk_offset, pm, pl, pacc);
}


/// Merge the per-block partials into the context vector.
///
/// One CTA per (head, query). The merge is the same online step the blocks ran,
/// applied across them, so splitting the scan changes nothing about the value.
__device__ void s_attn_combine(int cta, int nctas, int head,
                               const float *__restrict__ pm,
                               const float *__restrict__ pl,
                               const float *__restrict__ pacc, int nblock,
                               bf16 *__restrict__ ctx) {
    const int g = cta;
    const int d = int(threadIdx.x);
    // One thread per head dimension. A persistent grid runs this with the
    // block size the rest of the step uses, which is wider than DH, and a
    // thread past the end would write over whatever follows `ctx`.
    if (d >= DH) return;

    float m = -INFINITY, l = 0.0f, acc = 0.0f;
    for (int b = 0; b < nblock; ++b) {
        const size_t at = (size_t(b) * HKV + head) * GQA + g;
        const float bm = pm[at];
        if (bm == -INFINITY) continue;
        const float nm = fmaxf(m, bm);
        const float cr = (m == -INFINITY) ? 0.0f : __expf(m - nm);
        const float br = __expf(bm - nm);
        l = l * cr + pl[at] * br;
        acc = acc * cr + pacc[at * DH + d] * br;
        m = nm;
    }
    ctx[(size_t(head) * GQA + g) * DH + d] = __float2bfloat16(acc / l);
}

__global__ __launch_bounds__(DH) void k_attn_combine(
    const float *__restrict__ pm, const float *__restrict__ pl,
    const float *__restrict__ pacc, int nblock, bf16 *__restrict__ ctx) {
    s_attn_combine(int(blockIdx.x), int(gridDim.x), int(blockIdx.y), pm, pl,
                   pacc, nblock, ctx);
}


constexpr int QKV = QP + 2 * KVP;
constexpr int QKV_TILES = QKV / TILE_ROWS;
constexpr int O_TILES = H / TILE_ROWS;

/// Pre-norm, the fused Q/K/V projection, and this token's row of the cache.
///
/// `cache_update`, realised on the cache's own buffer: the row this token adds
/// is written where the scan will read it, so nothing is copied between steps.
__device__ void s_qkv(int cta, int nctas, const bf16 *__restrict__ h, const bf16 *__restrict__ gamma, const bf16 *__restrict__ w_qkv, bf16 *__restrict__ qkv, bf16 *__restrict__ k_tail, bf16 *__restrict__ v_tail, int cur_pos) {
    __shared__ float slots[WARPS];
    __shared__ __align__(16) bf16 xs[H];
    rms_norm_to_smem(h, gamma, xs, slots);
    ops::sync<ops::SyncKind::syncthreads>();
    int first, count;
    tile_span(QKV_TILES, cta, nctas, &first, &count);
    gemv_rows_direct<H>(w_qkv, xs, qkv, first * TILE_ROWS, count * TILE_ROWS);
    ops::sync<ops::SyncKind::syncthreads>();

    const int lo = first * TILE_ROWS, hi = (first + count) * TILE_ROWS;
    for (int row = lo + int(threadIdx.x); row < hi; row += THREADS) {
        if (row < QP) continue;
        const int off = row - QP;
        bf16 *dst = (off < KVP ? k_tail : v_tail) +
                    (size_t(cur_pos) * HKV * DH) + (off % KVP);
        *dst = qkv[row];
    }
}

__global__ __launch_bounds__(THREADS) void k_qkv(
    const bf16 *__restrict__ h, const bf16 *__restrict__ gamma,
    const bf16 *__restrict__ w_qkv, bf16 *__restrict__ qkv,
    bf16 *__restrict__ k_tail, bf16 *__restrict__ v_tail, int cur_pos) {
    s_qkv(int(blockIdx.x), int(gridDim.x), h, gamma, w_qkv, qkv, k_tail, v_tail, cur_pos);
}


/// The output projection and the residual add.
__device__ void s_o_proj(int cta, int nctas, const bf16 *__restrict__ ctx, const bf16 *__restrict__ w_o, const bf16 *__restrict__ h_in, bf16 *__restrict__ h_out) {
    __shared__ __align__(16) bf16 xs[QP];
    for (int i = int(threadIdx.x); i < QP; i += THREADS) xs[i] = ctx[i];
    ops::sync<ops::SyncKind::syncthreads>();
    int first, count;
    tile_span(O_TILES, cta, nctas, &first, &count);
    gemv_rows_direct<QP>(w_o, xs, h_out, first * TILE_ROWS, count * TILE_ROWS,
                         h_in);
}

__global__ __launch_bounds__(THREADS) void k_o_proj(
    const bf16 *__restrict__ ctx, const bf16 *__restrict__ w_o,
    const bf16 *__restrict__ h_in, bf16 *__restrict__ h_out) {
    s_o_proj(int(blockIdx.x), int(gridDim.x), ctx, w_o, h_in, h_out);
}


/// One full-attention layer.
std::vector<torch::Tensor> attn_layer(
    torch::Tensor h, torch::Tensor gamma, torch::Tensor w_qkv, torch::Tensor w_o,
    torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor k_tail,
    torch::Tensor v_tail, int64_t cur_pos) {
    check_bf16(h, H, "h");
    check_bf16(gamma, H, "gamma");
    check_bf16(w_qkv, int64_t(QKV) * H, "w_qkv");
    check_bf16(w_o, int64_t(H) * QP, "w_o");

    const int ctx_full = int(k_cache.numel() / (int64_t(HKV) * DH));
    const int ctx_tail = int(k_tail.numel() / (int64_t(HKV) * DH));
    TORCH_CHECK(cur_pos >= 0 && cur_pos < ctx_tail, "cur_pos out of the tail");

    const auto bf_opt = h.options();
    const auto f_opt = h.options().dtype(at::kFloat);
    auto qkv = torch::empty({QKV}, bf_opt);
    auto ctx = torch::empty({QP}, bf_opt);
    auto h_out = torch::empty({H}, bf_opt);

    const int nb_full = (ctx_full + ABLK - 1) / ABLK;
    // The tail holds `cur_pos + 1` live rows: this token's is the last written.
    const int live_tail = int(cur_pos) + 1;
    const int nb_tail = (live_tail + ABLK - 1) / ABLK;
    const int nblock = nb_full + nb_tail;

    auto pm = torch::empty({nblock, HKV, GQA}, f_opt);
    auto pl = torch::empty({nblock, HKV, GQA}, f_opt);
    auto pacc = torch::empty({nblock, HKV, GQA, DH}, f_opt);

    auto stream = at::cuda::getCurrentCUDAStream();
    const int ctas = sm_count();
    constexpr size_t kQkv = ProjSmem<H, IN_STAGES>::bytes();
    allow_smem(k_qkv, kQkv);

    k_qkv<<<ctas, THREADS, kQkv, stream>>>(
        static_cast<const bf16 *>(h.data_ptr()),
        static_cast<const bf16 *>(gamma.data_ptr()),
        static_cast<const bf16 *>(w_qkv.data_ptr()),
        static_cast<bf16 *>(qkv.data_ptr()),
        static_cast<bf16 *>(k_tail.data_ptr()),
        static_cast<bf16 *>(v_tail.data_ptr()), int(cur_pos));

    auto *Q = static_cast<bf16 *>(qkv.data_ptr());
    auto *PM = static_cast<float *>(pm.data_ptr());
    auto *PL = static_cast<float *>(pl.data_ptr());
    auto *PA = static_cast<float *>(pacc.data_ptr());
    if (nb_full)
        k_attn_block<<<dim3(nb_full, HKV), ATT_THREADS, 0, stream>>>(
            Q, static_cast<const bf16 *>(k_cache.data_ptr()),
            static_cast<const bf16 *>(v_cache.data_ptr()), ctx_full, 0, PM, PL, PA);
    if (nb_tail)
        k_attn_block<<<dim3(nb_tail, HKV), ATT_THREADS, 0, stream>>>(
            Q, static_cast<const bf16 *>(k_tail.data_ptr()),
            static_cast<const bf16 *>(v_tail.data_ptr()), live_tail, nb_full,
            PM, PL, PA);
    k_attn_combine<<<dim3(GQA, HKV), DH, 0, stream>>>(
        PM, PL, PA, nblock, static_cast<bf16 *>(ctx.data_ptr()));
    k_o_proj<<<ctas, THREADS, 0, stream>>>(
        static_cast<const bf16 *>(ctx.data_ptr()),
        static_cast<const bf16 *>(w_o.data_ptr()),
        static_cast<const bf16 *>(h.data_ptr()),
        static_cast<bf16 *>(h_out.data_ptr()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {h_out, qkv, ctx};
}


/// The closing norm and the head.
///
/// 131072 rows over 132 CTAs is 124 tiles each, which is where running tiles
/// ahead pays; the head is the single largest read in the step at 705 MB.
__device__ void s_head(char *arena, int cta, int nctas, const bf16 *__restrict__ h,
                       const bf16 *__restrict__ gf, const bf16 *__restrict__ whead,
                       int vocab, float *__restrict__ logits) {
    char *raw = arena;
    ProjSmem<H, IN_STAGES> sm(raw);
    rms_norm_to_smem(h, gf, sm.x, sm.slots);
    ops::sync<ops::SyncKind::syncthreads>();

    const int ntiles = vocab / TILE_ROWS;
    int first, count;
    tile_span(ntiles, cta, nctas, &first, &count);

    constexpr unsigned kBytes = unsigned(TILE_ROWS) * H * sizeof(bf16);
    constexpr int kTileElems = TILE_ROWS * H;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const bool elected = ops::shuffle_elect<THREADS>();
    if (elected)
        for (int s = 0; s < IN_STAGES; ++s) ops::mbarrier_init(&sm.bars[s], 1);
    ops::sync<ops::SyncKind::syncthreads>();
    for (int s = 0; s < IN_STAGES && s < count; ++s)
        if (elected) {
            ops::mbarrier_arrive_expect_tx(&sm.bars[s], kBytes);
            ops::tma_bulk_copy(whead + size_t(first + s) * kTileElems,
                               sm.stage + s * kTileElems, kTileElems, &sm.bars[s]);
        }
    for (int t = 0; t < count; ++t) {
        const int s = t % IN_STAGES;
        ops::mbarrier_wait_parity(&sm.bars[s], unsigned(t / IN_STAGES) & 1u);
        const bf16 *row = sm.stage + s * kTileElems + warp * H;
        float part[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        // `j` steps by 32, so `j & 7` is constant for a lane and would leave
        // seven accumulators at zero. The step index is what varies.
        for (int j = lane, u = 0; j < H; j += 32, ++u)
            part[u & 7] += ld(row + j) * ld(sm.x + j);
        float acc = (part[0] + part[1]) + (part[2] + part[3]) +
                    ((part[4] + part[5]) + (part[6] + part[7]));
        acc = ops::warp_reduce<ops::warp_sum>(acc);
        // `(fh @ w_head.t()).float()`: the product lands in bf16 and only then
        // widens, so the rounding is the matmul's, not the store's.
        if (lane == 0) logits[(first + t) * TILE_ROWS + warp] = bf(acc);
        ops::sync<ops::SyncKind::syncthreads>();
        const int next = t + IN_STAGES;
        if (next < count && elected) {
            ops::mbarrier_arrive_expect_tx(&sm.bars[s], kBytes);
            ops::tma_bulk_copy(whead + size_t(first + next) * kTileElems,
                               sm.stage + s * kTileElems, kTileElems, &sm.bars[s]);
        }
    }
    if (elected)
        for (int s = 0; s < IN_STAGES; ++s) ops::mbarrier_invalidate(&sm.bars[s]);
}

/// The arena every staged projection overlays, sized to the largest of them.
///
/// One kernel calls all of them, so one allocation has to cover the widest:
/// `s_out_proj` stages 4096-wide rows and needs more than `s_in_proj`'s 2688.
/// Getting this wrong is a shared-memory write past the end, which is what
/// compute-sanitizer reported before it was a constant.
constexpr size_t kMegaSmem = ProjSmem<H, IN_STAGES>::bytes();

/// Everything one step reads that does not change between steps.
struct Weights {
    const bf16 *win, *wout, *convw, *convb, *ggdn, *mscal;
    const bf16 *wqkv, *wo;
    const bf16 *wrt, *wup, *wdn, *wsu, *wsd;
    const float *eb;
    const bf16 *gam, *table, *whead, *gf;
};

/// Everything one step writes that is not a state it hands on.
struct Scratch {
    bf16 *h, *h2, *proj, *xbc, *y, *scan, *qkv, *ctx, *mid, *smid;
    float *acc, *rlog, *pm, *pl, *pacc;
    int *idx;
    float *gw;
    float *logits;
};

/// One decode step, resident.
///
/// The grid stays on the card for the whole step and a barrier is what ends a
/// stage, so nothing returns between the embedding and the head. Every
/// `grid.sync()` below sits where `model.py` reshards a value back out of its
/// mesh split -- an `tf.reshard(..., "gmem")` that closes a projection -- which
/// is a property of the authored dataflow rather than a choice made here.
///
/// The residual add is in place: the CTA that writes a row of the hidden vector
/// is the CTA that read it, so no barrier separates the two and no second pass
/// over 2688 elements is needed.
__global__ __launch_bounds__(THREADS) void mega_decode(
    Weights w, Scratch s, void **st, const int *kind, const int *at,
    const long *token_ids, int nlayer, int n_mamba, int n_attn, int cur_pos,
    int ctx_full, int ctx_tail, int vocab) {
    extern __shared__ __align__(16) char arena[];
    cg::grid_group grid = cg::this_grid();
    const int cta = int(blockIdx.x), nctas = int(gridDim.x);
    const int tid = int(threadIdx.x);

    const long tok = token_ids[0];
    for (int i = cta * THREADS + tid; i < H; i += nctas * THREADS)
        s.h[i] = w.table[size_t(tok) * H + i];
    grid.sync();

    const int live_tail = cur_pos + 1;
    const int nb_full = (ctx_full + ABLK - 1) / ABLK;
    const int nb_tail = (live_tail + ABLK - 1) / ABLK;

    for (int L = 0; L < nlayer; ++L) {
        const bf16 *gam = w.gam + size_t(L) * H;
        const int a = at[L];
        if (kind[L] == 0) {
            bf16 *conv_in = static_cast<bf16 *>(st[0 * n_mamba + a]);
            float *ssm_in = static_cast<float *>(st[1 * n_mamba + a]);
            bf16 *conv_out = static_cast<bf16 *>(st[2 * n_mamba + a]);
            float *ssm_out = static_cast<float *>(st[3 * n_mamba + a]);
            const bf16 *ms = w.mscal + size_t(a) * 3 * MH;
            s_in_proj(cta, nctas, s.h, gam, w.win + size_t(a) * PROJ * H,
                      s.proj);
            grid.sync();
            s_conv(cta, nctas, s.proj, conv_in, w.convw + size_t(a) * CONV * KER,
                   w.convb + size_t(a) * CONV, conv_out, s.xbc);
            grid.sync();
            s_ssm(cta, nctas, s.xbc, s.proj, ms, ssm_in, ssm_out, s.y);
            grid.sync();
            s_gate_norm(cta, nctas, s.y, s.xbc, s.proj, ms,
                        w.ggdn + size_t(a) * MI, s.scan);
            grid.sync();
            s_out_proj(cta, nctas, s.scan, w.wout + size_t(a) * H * MI, s.h,
                       s.h);
            grid.sync();
        } else if (kind[L] == 1) {
            const bf16 *kc = static_cast<const bf16 *>(st[4 * n_mamba + 0 * n_attn + a]);
            const bf16 *vc = static_cast<const bf16 *>(st[4 * n_mamba + 1 * n_attn + a]);
            bf16 *kt = static_cast<bf16 *>(st[4 * n_mamba + 2 * n_attn + a]);
            bf16 *vt = static_cast<bf16 *>(st[4 * n_mamba + 3 * n_attn + a]);
            s_qkv(cta, nctas, s.h, gam, w.wqkv + size_t(a) * QKV * H, s.qkv,
                  kt, vt, cur_pos);
            grid.sync();
            for (int b = cta; b < nb_full * HKV; b += nctas)
                s_attn_block(b / HKV, nb_full, b % HKV, s.qkv, kc, vc, ctx_full, 0,
                             s.pm, s.pl, s.pacc);
            for (int b = cta; b < nb_tail * HKV; b += nctas)
                s_attn_block(b / HKV, nb_tail, b % HKV, s.qkv, kt, vt, live_tail,
                             nb_full, s.pm, s.pl, s.pacc);
            grid.sync();
            for (int u = cta; u < GQA * HKV; u += nctas)
                s_attn_combine(u / HKV, 0, u % HKV, s.pm, s.pl, s.pacc,
                               nb_full + nb_tail, s.ctx);
            grid.sync();
            s_o_proj(cta, nctas, s.ctx, w.wo + size_t(a) * H * QP, s.h, s.h);
            grid.sync();
        } else {
            s_moe_logits(cta, nctas, s.h, gam, w.wrt + size_t(a) * E * H, s.h2,
                         s.rlog);
            grid.sync();
            if (cta == 0)
                s_moe_topk(0, 1, s.rlog, w.eb + size_t(a) * E, s.idx, s.gw);
            for (int i = cta * THREADS + tid; i < H; i += nctas * THREADS)
                s.acc[i] = 0.0f;
            grid.sync();
            for (int slot = 0; slot < KTOP; ++slot)
                s_moe_up(cta, nctas, slot, s.h2, w.wup + size_t(a) * E * I * H,
                         s.idx, s.mid);
            s_moe_shared_up(cta, nctas, s.h2, w.wsu + size_t(a) * IS * H, s.smid);
            grid.sync();
            for (int slot = 0; slot < KTOP; ++slot)
                s_moe_down(cta, nctas, slot, s.mid, w.wdn + size_t(a) * E * H * I,
                           s.idx, s.gw, s.acc);
            grid.sync();
            s_moe_finish(cta, nctas, s.smid, w.wsd + size_t(a) * H * IS, s.acc,
                         s.h, s.h);
            grid.sync();
        }
    }
    s_head(arena, cta, nctas, s.h, w.gf, w.whead, vocab, s.logits);
}

/// How many CTAs of `mega_decode` fit at once.
///
/// A cooperative launch may not ask for more than the card can hold resident:
/// its grid barrier deadlocks if any CTA is still queued. The answer depends on
/// the shared-memory request, so it is asked rather than assumed.
int mega_grid(size_t smem) {
    static int n = 0;
    if (n == 0) {
        // The ceiling has to be raised before residency is asked about it:
        // the default is 48 KB and the answer would be zero.
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            reinterpret_cast<const void *>(mega_decode),
            cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem)));
        cudaFuncAttributes fa{};
        C10_CUDA_CHECK(cudaFuncGetAttributes(
            &fa, reinterpret_cast<const void *>(mega_decode)));
        int per_sm = 0;
        C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &per_sm, reinterpret_cast<const void *>(mega_decode), THREADS, smem));
        TORCH_CHECK(per_sm > 0, "mega_decode is not resident: ", fa.sharedSizeBytes,
                    " bytes static plus ", smem,
                    " dynamic exceeds what an SM can hold, and a cooperative "
                    "launch deadlocks if any CTA stays queued");
        n = per_sm * sm_count();
    }
    return n;
}

/// One decode step as a single cooperative launch.
///
/// `weights` arrive in the pack's own order and `states` in the table order the
/// kernel indexes; both are views into tensors the twin already holds, so this
/// copies no weight.
torch::Tensor mega_step(std::vector<torch::Tensor> weights,
                        std::vector<torch::Tensor> states,
                        std::vector<torch::Tensor> scratch, torch::Tensor kinds,
                        torch::Tensor ats, torch::Tensor token_ids,
                        int64_t cur_pos, int64_t ctx_full, int64_t ctx_tail,
                        int64_t layers) {
    TORCH_CHECK(weights.size() == 18, "expected 18 packed weight tensors");
    TORCH_CHECK(scratch.size() == 18, "expected 18 scratch tensors");
    auto B = [&](const torch::Tensor &t) { return static_cast<bf16 *>(t.data_ptr()); };
    auto CB = [&](const torch::Tensor &t) {
        return static_cast<const bf16 *>(t.data_ptr());
    };
    auto F = [&](const torch::Tensor &t) { return static_cast<float *>(t.data_ptr()); };

    Weights w{CB(weights[0]),  CB(weights[1]),  CB(weights[2]),  CB(weights[3]),
              CB(weights[4]),  CB(weights[15]), CB(weights[5]),  CB(weights[6]),
              CB(weights[7]),  CB(weights[8]),  CB(weights[9]),  CB(weights[10]),
              CB(weights[11]), static_cast<const float *>(weights[17].data_ptr()),
              CB(weights[12]), CB(weights[13]), CB(weights[14]), CB(weights[16])};

    Scratch s{B(scratch[0]),  B(scratch[1]),  B(scratch[2]),  B(scratch[3]),
              B(scratch[4]),  B(scratch[5]),  B(scratch[6]),  B(scratch[7]),
              B(scratch[8]),  B(scratch[9]),  F(scratch[10]), F(scratch[11]),
              F(scratch[12]), F(scratch[13]), F(scratch[14]),
              static_cast<int *>(scratch[15].data_ptr()), F(scratch[16]),
              F(scratch[17])};

    // The state pointers travel as a device array, so the kernel indexes them by
    // (kind, layer) the way it indexes a weight.
    std::vector<void *> host_ptrs;
    host_ptrs.reserve(states.size());
    for (auto &t : states) host_ptrs.push_back(t.data_ptr());
    // The table has to outlive the launch. A local tensor is returned to the
    // caching allocator when this function returns, and the next allocation
    // hands the same bytes to somebody else while the kernel is still reading
    // them -- which reads as an illegal access one step later, not here.
    static torch::Tensor table;
    if (!table.defined() || table.numel() < int64_t(host_ptrs.size()))
        table = torch::empty({int64_t(host_ptrs.size())},
                             torch::dtype(torch::kInt64).device(token_ids.device()));
    C10_CUDA_CHECK(cudaMemcpyAsync(table.data_ptr(), host_ptrs.data(),
                                   host_ptrs.size() * sizeof(void *),
                                   cudaMemcpyHostToDevice,
                                   at::cuda::getCurrentCUDAStream()));

    // `layers` stops the walk early, which is how a divergence is bisected to
    // the layer that introduces it; 0 means the whole step.
    int nlayer = layers > 0 ? int(layers) : int(kinds.numel());
    int vocab = int(scratch[17].numel());
    int n_mamba = 0, n_attn = 0;
    auto kinds_cpu = kinds.to(torch::kCPU);
    const int *kp = kinds_cpu.data_ptr<int>();
    for (int i = 0; i < int(kinds.numel()); ++i) {
        if (kp[i] == 0) ++n_mamba;
        else if (kp[i] == 1) ++n_attn;
    }

    constexpr size_t kSmem = kMegaSmem;
    const int grid = mega_grid(kSmem);

    void **st = static_cast<void **>(table.data_ptr());
    int *kd = static_cast<int *>(kinds.data_ptr());
    int *atp = static_cast<int *>(ats.data_ptr());
    long *tok = static_cast<long *>(token_ids.data_ptr());
    int cp = int(cur_pos), cf = int(ctx_full), ct = int(ctx_tail);
    void *args[] = {&w,      &s,       &st,     &kd, &atp, &tok, &nlayer,
                    &n_mamba, &n_attn, &cp,     &cf, &ct,  &vocab};
    C10_CUDA_CHECK(cudaLaunchCooperativeKernel(
        reinterpret_cast<const void *>(mega_decode), dim3(grid), dim3(THREADS),
        args, kSmem, at::cuda::getCurrentCUDAStream()));
    return scratch[17];
}

/// How many CTAs the cooperative launch will use, for the twin to report.
int64_t mega_grid_size() { return mega_grid(kMegaSmem); }

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mamba_layer", &mamba_layer);
    m.def("moe_layer", &moe_layer);
    m.def("attn_layer", &attn_layer);
    m.def("mega_step", &mega_step);
    m.def("mega_grid_size", &mega_grid_size);
}
