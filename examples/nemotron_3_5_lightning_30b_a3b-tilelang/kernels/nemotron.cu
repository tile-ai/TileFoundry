/// Nemotron-3.5-Lightning-30B-A3B decode, handwritten over the runtime ops.
///
/// Every stage is a `__device__` function over gmem pointers plus a shared
/// scratch. The `__global__` wrappers below launch one stage at a time, which
/// is what makes a stage checkable on its own against the op-by-op path; a
/// persistent cooperative grid calls the same functions with a grid barrier
/// where a wrapper boundary is today.

/// Nothing here writes a thread index into arithmetic. A stage says which
/// layout it wants -- `row_tile`, `lane_vector`, `block_run` (shards.cuh) --
/// and `local()` hands each thread its slice; the tile operations are then
/// `ops::{tma_copy, copy, dot, mma, fill}`, which read the transfer width, the
/// contraction mesh and the tiling off those layouts. What stays written out is
/// the model's own arithmetic, including every place it rounds to bf16 mid
/// expression, because that is what the checkpoint does and it has to stay
/// readable.
#include <tilefoundry/runtime/cuda/runtime.cuh>

#include "shards.cuh"

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

/// The shared-memory run this kernel normalises over.
///
/// 2688 is ten and a half elements a thread, and a mesh either divides a run or
/// it does not; padding it to twelve costs 768 bytes of shared memory and lets
/// the sum of squares be one `ops::dot` over one shard instead of a shard plus
/// a remainder. The pad holds zeros, which a sum of squares ignores.
constexpr int HPAD = padded_span(H, THREADS);

/// One weight tile: ``TILE_ROWS`` whole rows of an ``(out, in)`` matrix.
///
/// A CTA owns a run of output rows, so its slice is a contiguous byte run and
/// the rank-1 bulk copy is the instruction that moves it. Eight rows is what
/// makes the run a whole number of 16-byte grains for both widths this model
/// uses, and it is one row per warp.
constexpr int TILE_ROWS = WARPS;

/// What a stage does with one row's product.
///
/// The product lands in bf16 before anything else touches it, because that is
/// what a bf16 matmul does; the epilogues differ only in what happens next.
/// `tile0` is where this call's tiles start, so an epilogue is told which tile
/// and which warp and never has to be handed the same offset twice.
struct store_bf16 {
    bf16 *out;
    int tile0;
    const bf16 *resid;
    __device__ void operator()(int tile, int warp, float acc) const {
        const int row = (tile0 + tile) * TILE_ROWS + warp;
        out[row] = __float2bfloat16(resid ? ld(resid + row) + bf(acc) : acc);
    }
};

/// `(fh @ w.t()).float()`: the product lands in bf16 and only then widens, so
/// the rounding is the matmul's and not the store's.
struct store_f32 {
    float *out;
    int tile0;
    __device__ void operator()(int tile, int warp, float acc) const {
        out[(tile0 + tile) * TILE_ROWS + warp] = bf(acc);
    }
};

/// The staged matrix-vector loop every projection in this file is.
///
/// `src(t)` names tile `t`'s first row -- which is where the expert index and
/// the row clamp live for the MoE and a plain stride for everything else --
/// `vec(t)` names the vector that tile contracts against, and `epi(tile, warp, acc)`
/// is what the stage does with a row's product.
///
/// The loads run `STAGES` tiles ahead, so the tile issued last is still in
/// flight while this one is folded. `ops::tma_copy` decides from the tile's
/// layout whether that is a bulk instruction and arrives on the barrier itself,
/// so the wait is the only thing this loop says about staging.
template <int IN, int STAGES, class Src, class Vec, class Epi>
__device__ void gemv_staged(Src src, Vec vec, int ntiles, bf16 *stage,
                            uint64_t *bars, Epi epi) {
    constexpr int kTileElems = TILE_ROWS * IN;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;

    if (ops::shuffle_elect<THREADS>())
        for (int s = 0; s < STAGES; ++s) ops::mbarrier_init(&bars[s], 1);
    ops::sync<ops::SyncKind::syncthreads>();

    auto issue = [&](int tile, int slot) {
        auto from = whole_run<kTileElems, THREADS>(
            cute::make_gmem_ptr(src(tile)));
        auto into = whole_run<kTileElems, THREADS>(
            cute::make_smem_ptr(stage + slot * kTileElems));
        ops::tma_copy(from, into, &bars[slot]);
    };
    for (int s = 0; s < STAGES && s < ntiles; ++s) issue(s, s);

    for (int t = 0; t < ntiles; ++t) {
        const int slot = t % STAGES;
        ops::mbarrier_wait_parity(&bars[slot], unsigned(t / STAGES) & 1u);

        auto rows = row_tile<IN, TILE_ROWS, THREADS>(
            cute::make_smem_ptr(stage + slot * kTileElems));
        auto xv = lane_vector<IN, THREADS>(cute::make_smem_ptr(vec(t)));
        float acc;
        auto out = cell(&acc);
        ops::dot(rows, xv, out);
        if (lane == 0) epi(t, warp, acc);

        ops::sync<ops::SyncKind::syncthreads>();
        const int next = t + STAGES;
        if (next < ntiles) issue(next, slot);
    }
    if (ops::shuffle_elect<THREADS>())
        for (int s = 0; s < STAGES; ++s) ops::mbarrier_invalidate(&bars[s]);
}

/// The same loop with the weights read where they lie.
///
/// A gemv reads each weight byte exactly once, so there is nothing for a shared
/// tile to be reused by. What staging costs instead is occupancy: three stages
/// of eight 2688-wide rows is 129 KB, which fits one CTA per SM and leaves the
/// memory system with one CTA's worth of requests in flight. Reading the rows
/// in place keeps the shared budget at zero, so the SM holds as many CTAs as
/// the launch offers.
///
/// Which of the two wins is a measurement, and both are here so it can be
/// repeated. Measured on one H200, one layer of the checkpoint:
///
///   Mamba-2 layer   staged 49.6 us   direct 85.7 us
///   MoE layer       staged  320 us   direct  114 us (with the router fixed)
///
/// The split is how many tiles a CTA gets. The Mamba projections give each CTA
/// about 78 rows, and running three tiles ahead hides the next load behind this
/// tile's fold. The MoE experts give each CTA under two tiles, so the prologue
/// issues more than the loop consumes and the staging is pure overhead.
template <int IN, class Src, class Vec, class Epi>
__device__ void gemv_direct(Src src, Vec vec, int ntiles, Epi epi) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    for (int t = 0; t < ntiles; ++t) {
        auto rows =
            row_tile<IN, TILE_ROWS, THREADS>(cute::make_gmem_ptr(src(t)));
        auto xv = lane_vector<IN, THREADS>(cute::make_smem_ptr(vec(t)));
        float acc;
        auto out = cell(&acc);
        ops::dot(rows, xv, out);
        if (lane == 0) epi(t, warp, acc);
    }
}

/// A run of tiles read straight through, and the vector every one of them
/// contracts against.
template <int IN> __device__ auto tile_stride(const bf16 *w, int tile0) {
    return [w, tile0](int t) {
        return w + size_t(tile0 + t) * TILE_ROWS * IN;
    };
}
__device__ auto same_vector(const bf16 *x) {
    return [x](int) { return x; };
}

/// The published ``NemotronHRMSNorm``, into shared memory.
///
/// The normalised value lands in bf16 **before** the weight multiplies it,
/// which is one rounding more than doing the whole thing in f32 would be. It is
/// what the checkpoint's own implementation does, so it is what this does.
///
/// `dst` holds `HPAD`, not `H`: the sum of squares runs over the padded shard
/// as one `ops::dot`, and the scale-and-weight pass that follows walks each
/// thread's own slice of it.
__device__ void rms_norm_to_smem(const bf16 *h, const bf16 *gamma, bf16 *dst,
                                 float *slots) {
    auto padded = block_run<HPAD, THREADS>(cute::make_smem_ptr(dst));
    ops::fill(padded, 0.0f, local_n(padded));
    ops::sync<ops::SyncKind::syncthreads>();
    copy_run<H, THREADS>(cute::make_gmem_ptr(h), cute::make_smem_ptr(dst));
    ops::sync<ops::SyncKind::syncthreads>();

    float sq;
    auto total = cell(&sq);
    auto ws = cute::make_tensor(cute::make_smem_ptr(slots), cute::Int<WARPS>{});
    ops::dot(padded, padded, total, ws);
    const float scale = rsqrtf(sq / float(H) + EPS);

    constexpr int SPAN = block_span(H, THREADS);
    auto xs = tilefoundry::local(
        block_run<SPAN, THREADS>(cute::make_smem_ptr(dst)));
    auto gs = tilefoundry::local(
        block_run<SPAN, THREADS>(cute::make_gmem_ptr(gamma)));
    const int n = int(cute::size(xs));
    for (int u = 0; u < n; ++u)
        xs(u) = __float2bfloat16(bf(ld(&xs(u)) * scale) * ld(&gs(u)));
    /// The 128 the mesh could not divide, one element a thread.
    const int tail = int(threadIdx.x);
    if (tail < H - SPAN)
        dst[SPAN + tail] = __float2bfloat16(bf(ld(dst + SPAN + tail) * scale) *
                                            ld(gamma + SPAN + tail));
}

/// The depthwise convolution over one chunk of channels, and the state it
/// hands on.
///
/// A chunk is a whole block's width, so which channel a thread owns is the
/// shard's answer and not an index this code computes. Every rounding below is
/// one the op-by-op reference makes: the products land in bf16, the fold over
/// the four taps is f32, and the bias and the silu each land again.
__device__ void mamba_conv_stage(const bf16 *col0, const bf16 *conv_state,
                                 const bf16 *conv_w, const bf16 *conv_b,
                                 bf16 *conv_out, bf16 *xbc, int chunk) {
    const bf16 *base = col0 + size_t(chunk) * THREADS;
    auto cols = tilefoundry::local(
        block_run<THREADS, THREADS>(cute::make_gmem_ptr(base)));
    const int n = int(cute::size(cols));

    for (int u = 0; u < n; ++u) {
        const int c = chunk * THREADS + index_of(cols, u, base);
        float win[KER];
        for (int k = 0; k < WIN; ++k) win[k] = ld(conv_state + c * WIN + k);
        win[WIN] = ld(&cols(u));
        for (int k = 0; k < WIN; ++k)
            conv_out[c * WIN + k] = __float2bfloat16(win[k + 1]);
        float acc = 0.0f;
        for (int k = 0; k < KER; ++k)
            acc += bf(win[k] * ld(conv_w + c * KER + k));
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
/// is over that run alone, and ``GRP`` is two elements a thread exactly, so the
/// fold is one `ops::dot` over the block. Every operand is sharded the same
/// way, so ``u`` names the same element in all of them and no index is
/// recomputed; only ``head``, which the model needs, is asked of the shard.
__device__ void mamba_gate_norm_stage(const bf16 *y, const bf16 *xbc,
                                      const bf16 *gate, const bf16 *mscal,
                                      const bf16 *ggdn, bf16 *scan, int group,
                                      float *slots, float *yf) {
    const int base = group * GRP;
    auto owned = block_run<GRP, THREADS>(cute::make_smem_ptr(yf));
    auto mine = tilefoundry::local(owned);
    auto yv = tilefoundry::local(
        block_run<GRP, THREADS>(cute::make_gmem_ptr(y + base)));
    auto xv = tilefoundry::local(
        block_run<GRP, THREADS>(cute::make_gmem_ptr(xbc + base)));
    auto gv = tilefoundry::local(
        block_run<GRP, THREADS>(cute::make_gmem_ptr(gate + base)));
    auto wv = tilefoundry::local(
        block_run<GRP, THREADS>(cute::make_gmem_ptr(ggdn + base)));
    auto sv = tilefoundry::local(
        block_run<GRP, THREADS>(cute::make_gmem_ptr(scan + base)));
    const int n = int(cute::size(mine));

    for (int u = 0; u < n; ++u) {
        const int head = (base + index_of(yv, u, y + base)) / MD;
        const float x = ld(&xv(u));
        const float d_skip = ld(mscal + 2 * MH + head);
        const float yd = bf(ld(&yv(u)) + bf(x * d_skip));
        mine(u) = yd * silu_f(ld(&gv(u)));
    }
    ops::sync<ops::SyncKind::syncthreads>();

    float sq;
    auto total = cell(&sq);
    auto ws = cute::make_tensor(cute::make_smem_ptr(slots), cute::Int<WARPS>{});
    ops::dot(owned, owned, total, ws);
    const float scale = rsqrtf(sq / float(GRP) + EPS);

    for (int u = 0; u < n; ++u)
        sv(u) = __float2bfloat16(bf(mine(u) * scale) * ld(&wv(u)));
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
///
/// The vector is `padded_span(IN, THREADS)` and not `IN`: a stage that
/// normalises into it reduces over a shard, and a shard's mesh either divides
/// the run or it does not.
template <int IN, int STAGES> struct ProjSmem {
    static constexpr int kVec = padded_span(IN, THREADS);
    bf16 *stage;
    bf16 *x;
    uint64_t *bars;
    float *slots;

    __device__ explicit ProjSmem(char *raw) {
        stage = reinterpret_cast<bf16 *>(raw);
        x = stage + size_t(STAGES) * TILE_ROWS * IN;
        bars = reinterpret_cast<uint64_t *>(x + kVec);
        slots = reinterpret_cast<float *>(bars + STAGES);
    }
    static constexpr size_t bytes() {
        return sizeof(bf16) * (size_t(STAGES) * TILE_ROWS * IN + kVec) +
               sizeof(uint64_t) * STAGES + sizeof(float) * WARPS;
    }
};

constexpr int IN_STAGES = 3;
constexpr int PROJ_TILES = PROJ / TILE_ROWS;
constexpr int OUT_TILES = H / TILE_ROWS;

/// Pre-norm and the input projection, in one launch.
///
/// Every CTA normalises the hidden row itself rather than one CTA doing it
/// behind a barrier: 2688 elements is not work worth dividing, and dividing it
/// would cost a barrier to put back together.
__device__ void s_in_proj(int cta, int nctas, const bf16 *h, const bf16 *gamma,
                          const bf16 *w_in, bf16 *proj) {
    __shared__ float slots[WARPS];
    __shared__ __align__(16) bf16 xs[HPAD];
    rms_norm_to_smem(h, gamma, xs, slots);
    ops::sync<ops::SyncKind::syncthreads>();
    int first, count;
    tile_span(PROJ_TILES, cta, nctas, &first, &count);
    gemv_direct<H>(tile_stride<H>(w_in, first), same_vector(xs), count,
                   store_bf16{proj, first, nullptr});
}

__global__ __launch_bounds__(THREADS) void k_in_proj(
    const bf16 *__restrict__ h, const bf16 *__restrict__ gamma,
    const bf16 *__restrict__ w_in, bf16 *__restrict__ proj) {
    s_in_proj(int(blockIdx.x), int(gridDim.x), h, gamma, w_in, proj);
}


/// The output projection and the residual add.
__device__ void s_out_proj(int cta, int nctas, const bf16 *scan,
                           const bf16 *w_out, const bf16 *h_in, bf16 *h_out) {
    __shared__ __align__(16) bf16 xs[MI];
    copy_run<MI, THREADS>(cute::make_gmem_ptr(scan), cute::make_smem_ptr(xs));
    ops::sync<ops::SyncKind::syncthreads>();
    int first, count;
    tile_span(OUT_TILES, cta, nctas, &first, &count);
    gemv_direct<MI>(tile_stride<MI>(w_out, first), same_vector(xs), count,
                    store_bf16{h_out, first, h_in});
}

__global__ __launch_bounds__(THREADS) void k_out_proj(
    const bf16 *__restrict__ scan, const bf16 *__restrict__ w_out,
    const bf16 *__restrict__ h_in, bf16 *__restrict__ h_out) {
    s_out_proj(int(blockIdx.x), int(gridDim.x), scan, w_out, h_in, h_out);
}


/// The channel run as whole-block chunks, one CTA taking every `nctas`-th.
///
/// A CTA's share of 6144 depends on the grid, and a shard's extents do not; a
/// chunk is a compile-time width and which chunks a CTA takes is the loop.
__device__ void s_conv(int cta, int nctas, const bf16 *proj,
                       const bf16 *conv_state, const bf16 *conv_w,
                       const bf16 *conv_b, bf16 *conv_out, bf16 *xbc) {
    constexpr int CHUNKS = CONV / THREADS;
    static_assert(CONV % THREADS == 0, "the channel run is whole blocks");
    for (int chunk = cta; chunk < CHUNKS; chunk += nctas)
        mamba_conv_stage(proj + MI, conv_state, conv_w, conv_b, conv_out, xbc,
                         chunk);
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

    /// Both projections read their weights where they lie, so neither asks for
    /// a dynamic allocation: one would cap how many CTAs an SM can hold for
    /// nothing in return.

    k_in_proj<<<ctas, THREADS, 0, stream>>>(
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
    k_out_proj<<<ctas, THREADS, 0, stream>>>(
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

/// The arena every stage that wants shared memory overlays, sized to the
/// largest of them.
///
/// One kernel calls all of them, so one allocation has to cover the widest.
/// Getting this wrong is a shared-memory write past the end, which is what
/// compute-sanitizer reported before it was a constant. Attention's tiles are
/// declared below, so the maximum is taken where both are in scope.
constexpr size_t kProjSmem = ProjSmem<H, IN_STAGES>::bytes();


/// The router's logits: 128 numbers, spread over the whole grid.
///
/// One warp per expert row, lanes striding it 16 bytes at a time. Written out
/// because the obvious shape -- one *thread* per expert, walking its row -- puts
/// consecutive threads 5376 bytes apart and coalesces nothing: measured at
/// 236 us for 688 KB, against 74 us for the 160 MB of expert weights beside it.
///
/// Every CTA normalises the hidden row itself, as elsewhere; CTA zero also
/// publishes it, because the expert projections read it next.
__device__ void s_moe_logits(int cta, int nctas, const bf16 *h,
                             const bf16 *gamma, const bf16 *w_router, bf16 *h2,
                             float *logits) {
    __shared__ float slots[WARPS];
    __shared__ __align__(16) bf16 xs[HPAD];

    rms_norm_to_smem(h, gamma, xs, slots);
    ops::sync<ops::SyncKind::syncthreads>();
    if (cta == 0)
        copy_run<H, THREADS>(cute::make_smem_ptr(xs), cute::make_gmem_ptr(h2));

    int first, count;
    tile_span(E / TILE_ROWS, cta, nctas, &first, &count);
    /// The router runs in f32 on both sides -- `h2.float() @ w.float().t()` --
    /// so this is the one projection that does not land in bf16 first.
    gemv_direct<H>(tile_stride<H>(w_router, first), same_vector(xs), count,
                   [logits, first](int tile, int warp, float acc) {
                       logits[(first + tile) * TILE_ROWS + warp] = acc;
                   });
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


/// All six routed experts' up projections, as one stream of tiles.
///
/// The six are not six pipelines. A CTA owns a run of output rows in every
/// expert, so its work is `KTOP * tiles_per_expert` tiles that differ only in
/// which expert's matrix they come from -- `t / tpe` picks the expert and
/// `t % tpe` the block within it. Run as one loop, the three-deep prefetch has
/// something to hide behind; run as six loops of under two tiles, the prologue
/// issues more than the loop consumes and the pipeline never starts. Measured:
/// 1.8 tiles a CTA reached 1.46 TB/s, and the head at 124 reaches 4.39.
///
/// This is the shape `mega_kernel.py` uses, where the same index carries the
/// expert and the block (`sel[t // 2]`, `(t % 2) * 8`).
__device__ void s_moe_up(char *arena, int cta, int nctas, const bf16 *h2,
                         const bf16 *w_up, const int *idx, bf16 *mid) {
    __shared__ __align__(16) bf16 xs[H];
    copy_run<H, THREADS>(cute::make_gmem_ptr(h2), cute::make_smem_ptr(xs));
    ops::sync<ops::SyncKind::syncthreads>();

    ProjSmem<H, MOE_STAGES> sm(arena);
    /// A CTA owns a run of output rows in every expert, so its work is
    /// `KTOP * count` tiles that differ only in which expert's matrix they come
    /// from: `tile / count` picks the expert and `tile % count` the block. The
    /// split is `tile_span`'s, so the runs are disjoint and cover the matrix --
    /// a per-CTA row count rounded up to whole tiles would overlap its
    /// neighbour, which a store survives and the atomic accumulation below does
    /// not.
    int first, count;
    tile_span(UP_TILES, cta, nctas, &first, &count);
    auto row_of = [first, count](int tile) {
        return (first + tile % count) * TILE_ROWS;
    };
    auto src = [w_up, idx, count, row_of](int tile) {
        return w_up + (size_t(idx[tile / count]) * I + row_of(tile)) * H;
    };
    gemv_staged<H, MOE_STAGES>(
        src, same_vector(xs), KTOP * count, sm.stage, sm.bars,
        [mid, count, row_of](int tile, int warp, float acc) {
            const float v = fmaxf(bf(acc), 0.0f);
            mid[size_t(tile / count) * I + row_of(tile) + warp] =
                __float2bfloat16(bf(v) * bf(v));
        });
}

__global__ __launch_bounds__(THREADS) void k_moe_up(
    const bf16 *__restrict__ h2, const bf16 *__restrict__ w_up,
    const int *__restrict__ idx, bf16 *__restrict__ mid) {
    extern __shared__ __align__(16) char arena[];
    s_moe_up(arena, int(blockIdx.x), int(gridDim.x), h2, w_up, idx, mid);
}

/// All six routed experts' down projections, as one stream of tiles.
///
/// Same shape as `s_moe_up`, over `(H, I)` instead of `(I, H)`, scaled by each
/// expert's routing weight and accumulated in f32. The accumulator is atomic
/// because six experts land on the same row.
__device__ void s_moe_down(char *arena, int cta, int nctas, const bf16 *mid,
                           const bf16 *w_down, const int *idx, const float *gw,
                           float *acc_out) {
    __shared__ __align__(16) bf16 xs[KTOP * I];
    copy_run<KTOP * I, THREADS>(cute::make_gmem_ptr(mid),
                                cute::make_smem_ptr(xs));
    ops::sync<ops::SyncKind::syncthreads>();

    ProjSmem<I, MOE_STAGES> sm(arena);
    /// Same stream of tiles as `s_moe_up`, over `(H, I)` instead of `(I, H)`.
    int first, count;
    tile_span(DOWN_TILES, cta, nctas, &first, &count);
    auto row_of = [first, count](int tile) {
        return (first + tile % count) * TILE_ROWS;
    };
    auto src = [w_down, idx, count, row_of](int tile) {
        return w_down + (size_t(idx[tile / count]) * H + row_of(tile)) * I;
    };
    /// Each expert contracts against its own slice of `mid`, so the vector is a
    /// function of the tile just as the matrix is.
    bf16 *slab = xs;
    auto vec = [slab, count](int tile) {
        return slab + size_t(tile / count) * I;
    };
    gemv_staged<I, MOE_STAGES>(
        src, vec, KTOP * count, sm.stage, sm.bars,
        [acc_out, gw, count, row_of](int tile, int warp, float dot) {
            atomicAdd(&acc_out[row_of(tile) + warp], bf(dot) * gw[tile / count]);
        });
}

__global__ __launch_bounds__(THREADS) void k_moe_down(
    const bf16 *__restrict__ mid, const bf16 *__restrict__ w_down,
    const int *__restrict__ idx, const float *__restrict__ gw,
    float *__restrict__ acc) {
    extern __shared__ __align__(16) char arena[];
    s_moe_down(arena, int(blockIdx.x), int(gridDim.x), mid, w_down, idx, gw, acc);
}

/// The shared expert's up projection. Every token goes through it, so it is not
/// selected and takes no routing weight.
__device__ void s_moe_shared_up(int cta, int nctas, const bf16 *__restrict__ h2, const bf16 *__restrict__ w_sh_up, bf16 *__restrict__ smid) {
    __shared__ __align__(16) bf16 xs[H];
    copy_run<H, THREADS>(cute::make_gmem_ptr(h2), cute::make_smem_ptr(xs));
    ops::sync<ops::SyncKind::syncthreads>();
    int first, count;
    tile_span(SH_UP_TILES, cta, nctas, &first, &count);
    /// relu squared, folded into the epilogue: the row is written once.
    gemv_direct<H>(tile_stride<H>(w_sh_up, first), same_vector(xs), count,
                   [smid, first](int tile, int warp, float acc) {
                       const float v = fmaxf(bf(acc), 0.0f);
                       smid[(first + tile) * TILE_ROWS + warp] =
                           __float2bfloat16(bf(v) * bf(v));
                   });
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
__device__ void s_moe_finish(int cta, int nctas, const bf16 *smid,
                             const bf16 *w_sh_down, const float *acc,
                             const bf16 *h_in, bf16 *h_out) {
    __shared__ __align__(16) bf16 xs[IS];
    copy_run<IS, THREADS>(cute::make_gmem_ptr(smid), cute::make_smem_ptr(xs));
    ops::sync<ops::SyncKind::syncthreads>();

    int first, count;
    tile_span(DOWN_TILES, cta, nctas, &first, &count);
    gemv_direct<IS>(tile_stride<IS>(w_sh_down, first), same_vector(xs), count,
                    [acc, h_in, h_out, first](int tile, int warp, float dot) {
                        const int row = (first + tile) * TILE_ROWS + warp;
                        /// The routed experts land in bf16 once, after all
                        /// six have summed in f32 -- `(total.to(bf16) + sh)`.
                        const float mix = bf(bf(acc[row]) + bf(dot));
                        h_out[row] = __float2bfloat16(ld(h_in + row) + mix);
                    });
}

__global__ __launch_bounds__(THREADS) void k_moe_finish(
    const bf16 *__restrict__ smid, const bf16 *__restrict__ w_sh_down,
    const float *__restrict__ acc, const bf16 *h_in, bf16 *h_out) {
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

    allow_smem(k_moe_up, kProjSmem);
    allow_smem(k_moe_down, kProjSmem);

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
    k_moe_up<<<ctas, THREADS, kProjSmem, stream>>>(
        H2, static_cast<const bf16 *>(w_up.data_ptr()), IDX, MID);
    k_moe_down<<<ctas, THREADS, kProjSmem, stream>>>(
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
///
/// 128 and not some other number: it is the tensor core's N for a `(16, 128)`
/// score block, so a block of keys is one `ops::mma`.
constexpr int ABLK = 128;
constexpr int ATT_THREADS = 256;

/// The arena one attention block overlays: the two key/value tiles first (they
/// are the mma's operands and want the widest alignment), then the probability
/// block, the query tile, and the per-query softmax state.
struct AttnSmem {
    bf16 *ks;
    bf16 *vs;
    bf16 *ps;
    bf16 *qs;
    float *sml;

    __device__ explicit AttnSmem(char *raw) {
        ks = reinterpret_cast<bf16 *>(raw);
        vs = ks + ABLK * DH;
        ps = vs + ABLK * DH;
        qs = ps + GQA * ABLK;
        sml = reinterpret_cast<float *>(qs + GQA * DH);
    }
    static constexpr size_t bytes() {
        return sizeof(bf16) * (2 * size_t(ABLK) * DH + GQA * ABLK + GQA * DH) +
               sizeof(float) * 3 * GQA;
    }
};

/// One block of keys, for one KV head, folded into a partial softmax.
///
/// The shape is the tilelang kernel's: a `(16, 128)` score block out of one
/// `ops::mma`, an online softmax over it with a query to sixteen adjacent
/// lanes, and a second `ops::mma` against the value block. Neither call names a
/// transpose -- `ks` is read as `(keys, dims)` and `vs` as `(dims, keys)`, and
/// that is a stride each.
__device__ void s_attn_block(char *arena, int cta, int nctas, int head,
                             const bf16 *q, const bf16 *kc, const bf16 *vc,
                             int nkey, int blk_offset, float *pm, float *pl,
                             float *pacc) {
    constexpr int GROUPS = ABLK / 8;
    AttnSmem sm(arena);
    const int blk = cta;
    const int tid = int(threadIdx.x);
    const int row = tid / GROUPS;
    const int lo = blk * ABLK;
    const int live = min(nkey - lo, ABLK);
    const size_t at = (size_t(blk_offset + blk) * HKV + head) * GQA;
    /// A block past the end still publishes its partial: the merge skips a
    /// block whose max is -inf, and uninitialised memory is not that.
    if (live <= 0) {
        if (tid < GQA) {
            pm[at + tid] = -INFINITY;
            pl[at + tid] = 0.0f;
        }
        return;
    }
    (void)nctas;

    /// The query rows this KV head owns, packed.
    {
        auto src = strided_tile<GQA, DH, DH, ATT_THREADS>(
            cute::make_gmem_ptr(q + size_t(head) * GQA * DH));
        auto dst = strided_tile<GQA, DH, DH, ATT_THREADS>(
            cute::make_smem_ptr(sm.qs));
        ops::copy(src, dst);
    }
    if (tid < GQA) {
        sm.sml[tid] = -INFINITY;
        sm.sml[GQA + tid] = 0.0f;
    }
    float out[ops::mma_acc_elems<GQA, DH, ATT_THREADS>()];
    for (int f = 0; f < int(sizeof(out) / sizeof(float)); ++f) out[f] = 0.0f;
    ops::sync<ops::SyncKind::syncthreads>();

    if (live == ABLK) {
        auto ksrc = strided_tile<ABLK, DH, HKV * DH, ATT_THREADS>(
            cute::make_gmem_ptr(kc + (size_t(lo) * HKV + head) * DH));
        auto kdst =
            strided_tile<ABLK, DH, DH, ATT_THREADS>(cute::make_smem_ptr(sm.ks));
        auto vsrc = strided_tile<ABLK, DH, HKV * DH, ATT_THREADS>(
            cute::make_gmem_ptr(vc + (size_t(lo) * HKV + head) * DH));
        auto vdst =
            strided_tile<ABLK, DH, DH, ATT_THREADS>(cute::make_smem_ptr(sm.vs));
        ops::copy(ksrc, kdst);
        ops::copy(vsrc, vdst);
    } else {
        /// The last block is shorter than the tile and the cache holds no rows
        /// past it, so each row is clamped to the last live one; its score is
        /// masked out below, so which row it read is moot. The clamp is the one
        /// thing a static shard cannot state -- the mapping is still the
        /// shard's, and only the row index is written here.
        auto kdst = tilefoundry::local(
            strided_tile<ABLK, DH, DH, ATT_THREADS>(cute::make_smem_ptr(sm.ks)));
        auto vdst = tilefoundry::local(
            strided_tile<ABLK, DH, DH, ATT_THREADS>(cute::make_smem_ptr(sm.vs)));
        const int m = int(cute::size(kdst));
        for (int u = 0; u < m; ++u) {
            const int idx = index_of(kdst, u, sm.ks);
            const int key = min(lo + idx / DH, lo + live - 1);
            const size_t src = (size_t(key) * HKV + head) * DH + idx % DH;
            kdst(u) = kc[src];
            vdst(u) = vc[src];
        }
    }
    ops::sync<ops::SyncKind::syncthreads>();

    /// The scores. `ks` is `(keys, dims)` and the contraction is over dims, so
    /// the `(N, K)` operand is that buffer read as it lies.
    float scores[ops::mma_acc_elems<GQA, ABLK, ATT_THREADS>()];
    auto sfg = ops::mma_acc_tensor<GQA, ABLK, ATT_THREADS>(
        cute::make_rmem_ptr(&scores[0]));
    ops::fill(sfg, 0.0f, int(sizeof(scores) / sizeof(float)));
    {
        auto qf = mma_operand<GQA, DH, DH, 1, ATT_THREADS>(
            cute::make_smem_ptr(sm.qs));
        auto kf = mma_operand<ABLK, DH, DH, 1, ATT_THREADS>(
            cute::make_smem_ptr(sm.ks));
        ops::mma(qf, kf, sfg);
    }
    for (int f = 0; f < int(sizeof(scores) / sizeof(float)); ++f) {
        auto rc = ops::mma_acc_coord<GQA, ABLK, ATT_THREADS>(f, tid);
        /// The product lands in bf16 before the scale, because
        /// `matmul(qg, kh.t())` is a bf16 matmul and only then `.float()`.
        sm.ps[cute::get<0>(rc) * ABLK + cute::get<1>(rc)] =
            __float2bfloat16(bf(scores[f]));
    }
    ops::sync<ops::SyncKind::syncthreads>();

    /// The online softmax, in place over the score block: a query to sixteen
    /// adjacent lanes and eight keys each, so both folds are butterflies over
    /// registers -- no shared round trip, and no lane idle.
    auto mine = tilefoundry::local(
        row_group_tile<GQA, ABLK, GROUPS, ATT_THREADS>(
            cute::make_smem_ptr(sm.ps)));
    const int n = int(cute::size(mine));
    const int key0 = index_of(mine, 0, sm.ps) - row * ABLK;

    float peak = -INFINITY;
    for (int u = 0; u < n; ++u)
        if (key0 + u < live) peak = fmaxf(peak, ld(&mine(u)) * QSCALE);
    peak = ops::warp_reduce<ops::warp_max, GROUPS>(peak);
    const float nm = fmaxf(sm.sml[row], peak);
    const float corr = (sm.sml[row] == -INFINITY) ? 0.0f : expf(sm.sml[row] - nm);

    float sum = 0.0f;
    for (int u = 0; u < n; ++u) {
        const float p =
            (key0 + u < live) ? expf(ld(&mine(u)) * QSCALE - nm) : 0.0f;
        mine(u) = __float2bfloat16(p);
        sum += p;
    }
    sum = ops::warp_reduce<ops::warp_sum, GROUPS>(sum);
    if (tid % GROUPS == 0) {
        sm.sml[GQA + row] = sm.sml[GQA + row] * corr + sum;
        sm.sml[row] = nm;
        sm.sml[2 * GQA + row] = corr;
    }
    ops::sync<ops::SyncKind::syncthreads>();

    /// Rescale what came before, then add this block's share. `vs` is
    /// `(keys, dims)` and the contraction is over keys, so the `(N, K)` operand
    /// is the same buffer with its strides the other way round.
    auto ofg = ops::mma_acc_tensor<GQA, DH, ATT_THREADS>(
        cute::make_rmem_ptr(&out[0]));
    for (int f = 0; f < int(sizeof(out) / sizeof(float)); ++f) {
        auto rc = ops::mma_acc_coord<GQA, DH, ATT_THREADS>(f, tid);
        out[f] *= sm.sml[2 * GQA + cute::get<0>(rc)];
    }
    {
        auto pf = mma_operand<GQA, ABLK, ABLK, 1, ATT_THREADS>(
            cute::make_smem_ptr(sm.ps));
        auto vf = mma_operand<DH, ABLK, 1, DH, ATT_THREADS>(
            cute::make_smem_ptr(sm.vs));
        ops::mma(pf, vf, ofg);
    }

    if (tid % GROUPS == 0) {
        pm[at + row] = sm.sml[row];
        pl[at + row] = sm.sml[GQA + row];
    }
    for (int f = 0; f < int(sizeof(out) / sizeof(float)); ++f) {
        auto rc = ops::mma_acc_coord<GQA, DH, ATT_THREADS>(f, tid);
        pacc[(at + cute::get<0>(rc)) * DH + cute::get<1>(rc)] = out[f];
    }
}

__global__ __launch_bounds__(ATT_THREADS) void k_attn_block(
    const bf16 *__restrict__ q, const bf16 *__restrict__ kc,
    const bf16 *__restrict__ vc, int nkey, int blk_offset,
    float *__restrict__ pm, float *__restrict__ pl, float *__restrict__ pacc) {
    extern __shared__ __align__(16) char arena[];
    s_attn_block(arena, int(blockIdx.x), int(gridDim.x), int(blockIdx.y), q, kc,
                 vc, nkey, blk_offset, pm, pl, pacc);
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


/// One launch runs the projections and the attention scan, so its arena is the
/// larger of the two shapes.
constexpr size_t kMegaSmem =
    kProjSmem > AttnSmem::bytes() ? kProjSmem : AttnSmem::bytes();

constexpr int QKV = QP + 2 * KVP;
constexpr int QKV_TILES = QKV / TILE_ROWS;
constexpr int O_TILES = H / TILE_ROWS;

/// Pre-norm, the fused Q/K/V projection, and this token's row of the cache.
///
/// `cache_update`, realised on the cache's own buffer: the row this token adds
/// is written where the scan will read it, so nothing is copied between steps.
__device__ void s_qkv(int cta, int nctas, const bf16 *__restrict__ h, const bf16 *__restrict__ gamma, const bf16 *__restrict__ w_qkv, bf16 *__restrict__ qkv, bf16 *__restrict__ k_tail, bf16 *__restrict__ v_tail, int cur_pos) {
    __shared__ float slots[WARPS];
    __shared__ __align__(16) bf16 xs[HPAD];
    rms_norm_to_smem(h, gamma, xs, slots);
    ops::sync<ops::SyncKind::syncthreads>();
    int first, count;
    tile_span(QKV_TILES, cta, nctas, &first, &count);
    gemv_direct<H>(tile_stride<H>(w_qkv, first), same_vector(xs), count,
                   store_bf16{qkv, first, nullptr});
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
    copy_run<QP, THREADS>(cute::make_gmem_ptr(ctx), cute::make_smem_ptr(xs));
    ops::sync<ops::SyncKind::syncthreads>();
    int first, count;
    tile_span(O_TILES, cta, nctas, &first, &count);
    gemv_direct<QP>(tile_stride<QP>(w_o, first), same_vector(xs), count,
                    store_bf16{h_out, first, h_in});
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
    allow_smem(k_attn_block, AttnSmem::bytes());
    k_qkv<<<ctas, THREADS, 0, stream>>>(
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
        k_attn_block<<<dim3(nb_full, HKV), ATT_THREADS, AttnSmem::bytes(), stream>>>(
            Q, static_cast<const bf16 *>(k_cache.data_ptr()),
            static_cast<const bf16 *>(v_cache.data_ptr()), ctx_full, 0, PM, PL, PA);
    if (nb_tail)
        k_attn_block<<<dim3(nb_tail, HKV), ATT_THREADS, AttnSmem::bytes(), stream>>>(
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
__device__ void s_head(char *arena, int cta, int nctas, const bf16 *h,
                       const bf16 *gf, const bf16 *whead, int vocab,
                       float *logits) {
    ProjSmem<H, IN_STAGES> sm(arena);
    rms_norm_to_smem(h, gf, sm.x, sm.slots);
    ops::sync<ops::SyncKind::syncthreads>();

    int first, count;
    tile_span(vocab / TILE_ROWS, cta, nctas, &first, &count);
    gemv_staged<H, IN_STAGES>(tile_stride<H>(whead, first), same_vector(sm.x),
                              count, sm.stage, sm.bars,
                              store_f32{logits, first});
}

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
    int ctx_full, int ctx_tail, int vocab, int skip) {
    /// `skip` turns individual stages off so their cost can be measured by
    /// difference. It is a measurement instrument, not a mode: every bit set
    /// produces wrong numbers on purpose. The barriers stay either way, so a
    /// difference is the stage's own work and not the barrier's.
    ///   1 in_proj   2 conv   4 ssm    8 gate_norm  16 out_proj
    ///  32 qkv      64 scan 128 combine 256 o_proj
    /// 512 logits 1024 topk 2048 up   4096 down   8192 shared_up 16384 finish
    /// 32768 head
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
            if (!(skip & 1))
                s_in_proj(cta, nctas, s.h, gam, w.win + size_t(a) * PROJ * H,
                      s.proj);
            grid.sync();
            if (!(skip & 2))
                s_conv(cta, nctas, s.proj, conv_in, w.convw + size_t(a) * CONV * KER,
                   w.convb + size_t(a) * CONV, conv_out, s.xbc);
            grid.sync();
            if (!(skip & 4))
                s_ssm(cta, nctas, s.xbc, s.proj, ms, ssm_in, ssm_out, s.y);
            grid.sync();
            if (!(skip & 8))
                s_gate_norm(cta, nctas, s.y, s.xbc, s.proj, ms,
                        w.ggdn + size_t(a) * MI, s.scan);
            grid.sync();
            if (!(skip & 16))
                s_out_proj(cta, nctas, s.scan, w.wout + size_t(a) * H * MI, s.h,
                       s.h);
            grid.sync();
        } else if (kind[L] == 1) {
            const bf16 *kc = static_cast<const bf16 *>(st[4 * n_mamba + 0 * n_attn + a]);
            const bf16 *vc = static_cast<const bf16 *>(st[4 * n_mamba + 1 * n_attn + a]);
            bf16 *kt = static_cast<bf16 *>(st[4 * n_mamba + 2 * n_attn + a]);
            bf16 *vt = static_cast<bf16 *>(st[4 * n_mamba + 3 * n_attn + a]);
            if (!(skip & 32))
                s_qkv(cta, nctas, s.h, gam, w.wqkv + size_t(a) * QKV * H, s.qkv,
                  kt, vt, cur_pos);
            grid.sync();
            if (!(skip & 64))
            for (int b = cta; b < nb_full * HKV; b += nctas)
                s_attn_block(arena, b / HKV, nb_full, b % HKV, s.qkv, kc, vc,
                             ctx_full, 0, s.pm, s.pl, s.pacc);
            if (!(skip & 64))
            for (int b = cta; b < nb_tail * HKV; b += nctas)
                s_attn_block(arena, b / HKV, nb_tail, b % HKV, s.qkv, kt, vt,
                             live_tail, nb_full, s.pm, s.pl, s.pacc);
            grid.sync();
            if (!(skip & 128))
            for (int u = cta; u < GQA * HKV; u += nctas)
                s_attn_combine(u / HKV, 0, u % HKV, s.pm, s.pl, s.pacc,
                               nb_full + nb_tail, s.ctx);
            grid.sync();
            if (!(skip & 256))
                s_o_proj(cta, nctas, s.ctx, w.wo + size_t(a) * H * QP, s.h, s.h);
            grid.sync();
        } else {
            if (!(skip & 512))
                s_moe_logits(cta, nctas, s.h, gam, w.wrt + size_t(a) * E * H, s.h2,
                         s.rlog);
            grid.sync();
            if (cta == 0 && !(skip & 1024))
                s_moe_topk(0, 1, s.rlog, w.eb + size_t(a) * E, s.idx, s.gw);
            for (int i = cta * THREADS + tid; i < H; i += nctas * THREADS)
                s.acc[i] = 0.0f;
            grid.sync();
            if (!(skip & 2048))
                s_moe_up(arena, cta, nctas, s.h2,
                         w.wup + size_t(a) * E * I * H, s.idx, s.mid);
            if (!(skip & 8192))
                s_moe_shared_up(cta, nctas, s.h2, w.wsu + size_t(a) * IS * H, s.smid);
            grid.sync();
            if (!(skip & 4096))
                s_moe_down(arena, cta, nctas, s.mid,
                           w.wdn + size_t(a) * E * H * I, s.idx, s.gw, s.acc);
            grid.sync();
            if (!(skip & 16384))
                s_moe_finish(cta, nctas, s.smid, w.wsd + size_t(a) * H * IS, s.acc,
                         s.h, s.h);
            grid.sync();
        }
    }
    if (!(skip & 32768))
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
                        int64_t layers, int64_t skip) {
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
    int sk = int(skip);
    void *args[] = {&w,      &s,       &st,     &kd, &atp, &tok, &nlayer,
                    &n_mamba, &n_attn, &cp,     &cf, &ct,  &vocab, &sk};
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
