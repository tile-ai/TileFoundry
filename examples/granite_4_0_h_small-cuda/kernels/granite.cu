// Granite-4.0-H-Small decode kernels, CUDA C.
//
// Batch one, one token per step. That single fact sets every shape here: every
// matmul is a matvec, so the whole step is bandwidth over the weights and
// nothing else. The design follows from it:
//
//   * every weight is read exactly once per step, in its checkpoint (out, in)
//     orientation, so a warp streams one contiguous output row;
//   * one warp owns ROWS output rows at a time, which puts ROWS independent
//     16-byte loads in flight per lane -- enough memory-level parallelism to
//     keep HBM busy at the low occupancy a matvec allows;
//   * the shared input vector is staged in shared memory once per block;
//   * cheap elementwise work (residual, SiLU gate, the routing mix) rides in a
//     matvec's epilogue rather than in a kernel of its own.
//
// Accumulation is f32 everywhere, rounded to bf16 once at each value the model
// itself stores as bf16, which is what torch's bf16 matmul does.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <cstdint>

namespace {

constexpr int WARP = 32;

__device__ __forceinline__ float bf2f(__nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ __nv_bfloat16 f2bf(float v) { return __float2bfloat16(v); }

__device__ __forceinline__ float silu_f(float v) { return v / (1.0f + __expf(-v)); }

// torch's bf16 SiLU: compute in f32 from the rounded input, round the result.
__device__ __forceinline__ __nv_bfloat16 silu_bf(float v) {
  return f2bf(silu_f(v));
}

__device__ __forceinline__ float warp_all_reduce(float v) {
#pragma unroll
  for (int off = WARP / 2; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
  return v;
}

__device__ __forceinline__ float warp_reduce(float v) {
#pragma unroll
  for (int off = WARP / 2; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
  return v;
}

// Accumulate the eight bf16 lanes of one 16-byte pair into *acc*.
__device__ __forceinline__ void fma8(float &acc, const uint4 &w, const uint4 &x) {
  const __nv_bfloat162 *wp = reinterpret_cast<const __nv_bfloat162 *>(&w);
  const __nv_bfloat162 *xp = reinterpret_cast<const __nv_bfloat162 *>(&x);
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    float2 a = __bfloat1622float2(wp[j]);
    float2 b = __bfloat1622float2(xp[j]);
    acc = fmaf(a.x, b.x, acc);
    acc = fmaf(a.y, b.y, acc);
  }
}

// ---------------------------------------------------------------------------
// The matvec family.
//
// One warp owns ROWS rows of W and walks them together, so the x vector staged
// in shared memory is read once for ROWS weight streams. The epilogue is what
// differs between the members; the loop above it is identical, which is why it
// lives in one device function.
//
// ROWS * UNROLL is the number of 16-byte loads a lane has outstanding, and that
// product -- not occupancy -- is what decides whether HBM stays busy. A matvec
// has one thread per output row and no reuse to hide behind, so a shape with
// few rows (an 8192->4096 projection is 128 blocks on 132 SMs) cannot get its
// parallelism from the grid; it has to come from the loop. UNROLL is why: four
// steps of the reduction issued together turn 2 MB of in-flight reads into 8.
// ---------------------------------------------------------------------------

template <int ROWS, int UNROLL>
__device__ __forceinline__ void rows_dot(const __nv_bfloat16 *__restrict__ w_base, int len,
                                         int stride, const uint4 *__restrict__ xs, int lane,
                                         int rows_live, float *acc) {
#pragma unroll
  for (int r = 0; r < ROWS; ++r) acc[r] = 0.0f;
  const int nv = len >> 3;
  const uint4 *w[ROWS];
#pragma unroll
  for (int r = 0; r < ROWS; ++r) {
    const int row = r < rows_live ? r : 0;
    w[r] = reinterpret_cast<const uint4 *>(w_base + (size_t)row * stride);
  }
  // A whole number of unrolled steps first, so every lane takes the same trip
  // count and the warp does not diverge inside the loop.
  const int full = (nv / (UNROLL * WARP)) * (UNROLL * WARP);
  for (int i = lane; i < full; i += UNROLL * WARP) {
    uint4 xv[UNROLL], wv[ROWS][UNROLL];
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) xv[u] = xs[i + u * WARP];
#pragma unroll
    for (int r = 0; r < ROWS; ++r)
#pragma unroll
      for (int u = 0; u < UNROLL; ++u) wv[r][u] = w[r][i + u * WARP];
#pragma unroll
    for (int r = 0; r < ROWS; ++r)
#pragma unroll
      for (int u = 0; u < UNROLL; ++u) fma8(acc[r], wv[r][u], xv[u]);
  }
  for (int i = full + lane; i < nv; i += WARP) {
    const uint4 xv = xs[i];
    uint4 wv[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) wv[r] = w[r][i];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) fma8(acc[r], wv[r], xv);
  }
#pragma unroll
  for (int r = 0; r < ROWS; ++r) acc[r] = warp_reduce(acc[r]);
}

// Stage the shared input vector into shared memory.
__device__ __forceinline__ void stage_x(const __nv_bfloat16 *__restrict__ x, uint4 *xs, int n) {
  const int nv = n >> 3;
  const uint4 *xg = reinterpret_cast<const uint4 *>(x);
  for (int i = threadIdx.x; i < nv; i += blockDim.x) xs[i] = xg[i];
  __syncthreads();
}

// FP32_OUT picks the epilogue: bf16 as the model stores it, or a scaled f32
// (which only the head wants, so its logits keep full precision for sampling).
template <int ROWS, int NWARP, int UNROLL, int FP32_OUT>
__global__ __launch_bounds__(NWARP *WARP) void gemv_kernel(
    const __nv_bfloat16 *__restrict__ w, const __nv_bfloat16 *__restrict__ x,
    void *__restrict__ y, int m, int n, float alpha) {
  extern __shared__ uint4 xs[];
  stage_x(x, xs, n);

  const int lane = threadIdx.x & (WARP - 1);
  const int warp = threadIdx.x >> 5;
  const int row0 = (blockIdx.x * NWARP + warp) * ROWS;
  if (row0 >= m) return;
  const int live = min(ROWS, m - row0);

  float acc[ROWS];
  rows_dot<ROWS, UNROLL>(w + (size_t)row0 * n, n, n, xs, lane, live, acc);

  if (lane == 0) {
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      if (r >= live) break;
      if (FP32_OUT) static_cast<float *>(y)[row0 + r] = acc[r] * alpha;
      else static_cast<__nv_bfloat16 *>(y)[row0 + r] = f2bf(acc[r]);
    }
  }
}

// `residual + block * residual_multiplier`, the one arithmetic step the layer
// itself owns.
__global__ void residual_add_kernel(const __nv_bfloat16 *__restrict__ a,
                                    const __nv_bfloat16 *__restrict__ b,
                                    __nv_bfloat16 *__restrict__ out, int n, float mult) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  out[i] = f2bf(bf2f(a[i]) + bf2f(f2bf(mult * bf2f(b[i]))));
}

// One position of the key or value cache. The slot index is runtime data on
// the device, so a captured graph replays without asking the host where to
// write.
__global__ void cache_write_kernel(__nv_bfloat16 *__restrict__ cache,
                                   const __nv_bfloat16 *__restrict__ entry,
                                   const int *__restrict__ cur_pos, int width) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= width) return;
  cache[(size_t)(*cur_pos) * width + i] = entry[i];
}

// ---------------------------------------------------------------------------
// Norms
// ---------------------------------------------------------------------------

// One block, whole vector. n is 4096 or 8192 here, so eight elements per thread
// at 512 threads covers both without a grid-stride loop over global memory.
template <int NTHREAD>
__global__ __launch_bounds__(NTHREAD) void rms_norm_kernel(
    const __nv_bfloat16 *__restrict__ x, const __nv_bfloat16 *__restrict__ w,
    __nv_bfloat16 *__restrict__ y, int n, float eps) {
  __shared__ float red[NTHREAD / WARP];
  const uint4 *xv = reinterpret_cast<const uint4 *>(x);
  const int nv = n >> 3;

  float ss = 0.0f;
  for (int i = threadIdx.x; i < nv; i += NTHREAD) {
    uint4 v = xv[i];
    const __nv_bfloat162 *p = reinterpret_cast<const __nv_bfloat162 *>(&v);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      float2 f = __bfloat1622float2(p[j]);
      ss = fmaf(f.x, f.x, ss);
      ss = fmaf(f.y, f.y, ss);
    }
  }
  ss = warp_all_reduce(ss);
  const int lane = threadIdx.x & (WARP - 1);
  const int warp = threadIdx.x >> 5;
  if (lane == 0) red[warp] = ss;
  __syncthreads();
  if (threadIdx.x < NTHREAD / WARP) ss = red[threadIdx.x];
  else ss = 0.0f;
  if (warp == 0) {
    ss = warp_all_reduce(ss);
    if (lane == 0) red[0] = ss;
  }
  __syncthreads();
  const float scale = rsqrtf(red[0] / (float)n + eps);

  const uint4 *wv = reinterpret_cast<const uint4 *>(w);
  uint4 *yv = reinterpret_cast<uint4 *>(y);
  for (int i = threadIdx.x; i < nv; i += NTHREAD) {
    uint4 xr = xv[i], wr = wv[i], out;
    const __nv_bfloat162 *xp = reinterpret_cast<const __nv_bfloat162 *>(&xr);
    const __nv_bfloat162 *wp = reinterpret_cast<const __nv_bfloat162 *>(&wr);
    __nv_bfloat162 *op = reinterpret_cast<__nv_bfloat162 *>(&out);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      float2 f = __bfloat1622float2(xp[j]);
      float2 g = __bfloat1622float2(wp[j]);
      op[j] = __floats2bfloat162_rn(f.x * scale * g.x, f.y * scale * g.y);
    }
    yv[i] = out;
  }
}

// `GraniteMoeHybridRMSNormGated`: the gate multiplies *before* the norm, so it
// changes the norm rather than only scaling its result.
template <int NTHREAD>
__global__ __launch_bounds__(NTHREAD) void rms_norm_gated_kernel(
    const __nv_bfloat16 *__restrict__ x, const __nv_bfloat16 *__restrict__ gate,
    const __nv_bfloat16 *__restrict__ w, __nv_bfloat16 *__restrict__ y, int n, float eps) {
  __shared__ float red[NTHREAD / WARP];
  extern __shared__ float tbuf[];  // n floats: the gated value, reused below

  const uint4 *xv = reinterpret_cast<const uint4 *>(x);
  const uint4 *gv = reinterpret_cast<const uint4 *>(gate);
  const int nv = n >> 3;

  float ss = 0.0f;
  for (int i = threadIdx.x; i < nv; i += NTHREAD) {
    uint4 xr = xv[i], gr = gv[i];
    const __nv_bfloat162 *xp = reinterpret_cast<const __nv_bfloat162 *>(&xr);
    const __nv_bfloat162 *gp = reinterpret_cast<const __nv_bfloat162 *>(&gr);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      float2 f = __bfloat1622float2(xp[j]);
      float2 g = __bfloat1622float2(gp[j]);
      float t0 = f.x * silu_f(g.x);
      float t1 = f.y * silu_f(g.y);
      tbuf[i * 8 + j * 2 + 0] = t0;
      tbuf[i * 8 + j * 2 + 1] = t1;
      ss = fmaf(t0, t0, ss);
      ss = fmaf(t1, t1, ss);
    }
  }
  ss = warp_all_reduce(ss);
  const int lane = threadIdx.x & (WARP - 1);
  const int warp = threadIdx.x >> 5;
  if (lane == 0) red[warp] = ss;
  __syncthreads();
  ss = (threadIdx.x < NTHREAD / WARP) ? red[threadIdx.x] : 0.0f;
  if (warp == 0) {
    ss = warp_all_reduce(ss);
    if (lane == 0) red[0] = ss;
  }
  __syncthreads();
  const float scale = rsqrtf(red[0] / (float)n + eps);

  const uint4 *wv = reinterpret_cast<const uint4 *>(w);
  uint4 *yv = reinterpret_cast<uint4 *>(y);
  for (int i = threadIdx.x; i < nv; i += NTHREAD) {
    uint4 wr = wv[i], out;
    const __nv_bfloat162 *wp = reinterpret_cast<const __nv_bfloat162 *>(&wr);
    __nv_bfloat162 *op = reinterpret_cast<__nv_bfloat162 *>(&out);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      float2 g = __bfloat1622float2(wp[j]);
      op[j] = __floats2bfloat162_rn(tbuf[i * 8 + j * 2 + 0] * scale * g.x,
                                    tbuf[i * 8 + j * 2 + 1] * scale * g.y);
    }
    yv[i] = out;
  }
}

// ---------------------------------------------------------------------------
// Mamba-2 mixer
// ---------------------------------------------------------------------------

// The depthwise causal convolution at one token. The window closes on this
// token, so there is one multiply against the kernel and one reduction over it;
// channels never mix. The window slides here too: a thread owns one channel, so
// reading its three stored columns before writing the next three is safe
// in place.
__global__ void mamba_conv_kernel(__nv_bfloat16 *__restrict__ state,    // [C, W]
                                  const __nv_bfloat16 *__restrict__ col,  // [C]
                                  const __nv_bfloat16 *__restrict__ w,    // [C, K]
                                  const __nv_bfloat16 *__restrict__ b,    // [C]
                                  __nv_bfloat16 *__restrict__ out,        // [C]
                                  int channels, int win, int krn) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= channels) return;

  const __nv_bfloat16 entry = col[c];
  __nv_bfloat16 *st = state + (size_t)c * win;
  const __nv_bfloat16 *wk = w + (size_t)c * krn;

  // The products are rounded before they are summed: HF forms the elementwise
  // `window * weight` at bf16 and only then reduces it.
  float acc = 0.0f;
  __nv_bfloat16 hist[8];
#pragma unroll 4
  for (int k = 0; k < win; ++k) hist[k] = st[k];
  for (int k = 0; k < win; ++k) acc += bf2f(f2bf(bf2f(wk[k]) * bf2f(hist[k])));
  acc += bf2f(f2bf(bf2f(wk[win]) * bf2f(entry)));

  // Round the reduction, add the bias, round, then activate: the same three
  // landings torch makes for `sum(...) + bias` followed by SiLU at bf16.
  const float summed = bf2f(f2bf(acc)) + bf2f(b[c]);
  out[c] = silu_bf(bf2f(f2bf(summed)));

  for (int k = 0; k + 1 < win; ++k) st[k] = hist[k + 1];
  st[win - 1] = entry;
}

// One token of the selective state-space recurrence, one warp per (head, row).
//
// The state is [heads, head_dim, state] f32 with `state` innermost, so a warp's
// 128 f32 are 512 contiguous bytes -- read once, updated, written back once.
// That traffic (4 MB per layer each way) is the only thing here that is not
// register-resident.
template <int NWARP>
__global__ __launch_bounds__(NWARP *WARP) void mamba_ssm_kernel(
    float *__restrict__ ssm,              // [NH, PD, NS]
    const __nv_bfloat16 *__restrict__ x,  // [NH * PD]
    const __nv_bfloat16 *__restrict__ b_vec, const __nv_bfloat16 *__restrict__ c_vec,  // [NS]
    const __nv_bfloat16 *__restrict__ dt_raw_v,  // [NH]
    const __nv_bfloat16 *__restrict__ a_log, const __nv_bfloat16 *__restrict__ dt_bias,
    const __nv_bfloat16 *__restrict__ d_skip,
    __nv_bfloat16 *__restrict__ y,  // [NH * PD]
    int nh, int pd, int ns) {
  extern __shared__ float bc[];  // 2 * ns floats: B then C
  const int exp_dim = nh * pd;
  for (int i = threadIdx.x; i < ns; i += blockDim.x) {
    bc[i] = bf2f(b_vec[i]);
    bc[ns + i] = bf2f(c_vec[i]);
  }
  __syncthreads();

  const int lane = threadIdx.x & (WARP - 1);
  const int warp = threadIdx.x >> 5;
  const int pair = blockIdx.x * NWARP + warp;  // one (head, row) per warp
  if (pair >= exp_dim) return;
  const int h = pair / pd;

  // dt is per head: softplus(dt_raw + dt_bias) at bf16, as HF forms it, and the
  // decay is exp(dt * -exp(A_log)) in f32.
  const float dt_raw = bf2f(f2bf(bf2f(dt_raw_v[h]) + bf2f(dt_bias[h])));
  const float dt = bf2f(f2bf(log1pf(__expf(-fabsf(dt_raw))) + fmaxf(dt_raw, 0.0f)));
  const float decay = __expf(dt * -__expf(bf2f(a_log[h])));
  const float xv = bf2f(x[pair]);

  float *row = ssm + (size_t)pair * ns;
  float acc = 0.0f;
  for (int i = lane * 4; i < ns; i += WARP * 4) {
    float4 s = *reinterpret_cast<const float4 *>(row + i);
    float nb[4] = {bc[i], bc[i + 1], bc[i + 2], bc[i + 3]};
    float nc[4] = {bc[ns + i], bc[ns + i + 1], bc[ns + i + 2], bc[ns + i + 3]};
    float *sp = reinterpret_cast<float *>(&s);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      // dB then dBx, in that order and at bf16: HF discretises B first and only
      // then scales it by the token, and the two orders differ at bf16.
      const float dbx = bf2f(f2bf(bf2f(f2bf(dt * nb[j])) * xv));
      sp[j] = fmaf(sp[j], decay, dbx);
      // The read-out rounds the state to bf16 and accumulates in f32, which is
      // what the batched matvec HF calls does.
      acc = fmaf(bf2f(f2bf(sp[j])), nc[j], acc);
    }
    *reinterpret_cast<float4 *>(row + i) = s;
  }
  acc = warp_reduce(acc);
  if (lane == 0)
    y[pair] = f2bf(bf2f(f2bf(acc)) + bf2f(f2bf(xv * bf2f(d_skip[h]))));
}

// ---------------------------------------------------------------------------
// Attention (no positional encoding: this checkpoint is `nope`)
// ---------------------------------------------------------------------------

// Online-softmax decode attention, split over positions as well as heads.
//
// There are only 32 query heads, so a block per head leaves three quarters of
// the machine idle and makes every warp walk the whole cache one dependent load
// at a time -- 94 us at 2048 positions for 8 MB of reads. SPLITS stripes the
// positions across that many more blocks, each finishing a partial softmax
// (running max, running sum, weighted accumulator) that `attn_combine_kernel`
// merges. The merge is exact: rescaling two partials against their joint max is
// what the online softmax already does at every step.
//
// Each lane owns PER_LANE consecutive head dimensions, so a warp's read of one
// position is 256 contiguous bytes. Key and value are loaded together: the
// value is not needed until after the dot product, but issuing both up front
// costs nothing and removes a dependent stall from the inner loop.
template <int NWARP, int HD, int SPLITS>
__global__ __launch_bounds__(NWARP *WARP) void attn_decode_kernel(
    const __nv_bfloat16 *__restrict__ q,        // [HQ, HD]
    const __nv_bfloat16 *__restrict__ k_cache,  // [CAP, HKV, HD]
    const __nv_bfloat16 *__restrict__ v_cache,
    const int *__restrict__ cur_pos,
    float *__restrict__ part_acc,  // [HQ, SPLITS, HD]
    float *__restrict__ part_ms,   // [HQ, SPLITS, 2]
    int hkv, int groups, float scale) {
  constexpr int PER_LANE = HD / WARP;  // 4 for head_dim 128
  __shared__ float red_m[NWARP], red_s[NWARP];
  __shared__ float red_acc[NWARP][HD];

  const int h = blockIdx.x;
  const int split = blockIdx.y;
  const int kvh = h / groups;
  const int lane = threadIdx.x & (WARP - 1);
  const int warp = threadIdx.x >> 5;
  const int len = *cur_pos + 1;
  const int kv_stride = hkv * HD;

  // Contiguous stripes, so a split's reads stay together in memory. A split
  // past the live length simply contributes an empty partial.
  const int chunk = (len + SPLITS - 1) / SPLITS;
  const int begin = split * chunk;
  const int end = min(begin + chunk, len);

  float qr[PER_LANE];
#pragma unroll
  for (int j = 0; j < PER_LANE; ++j) qr[j] = bf2f(q[h * HD + lane * PER_LANE + j]);

  float m = -INFINITY, s = 0.0f;
  float acc[PER_LANE];
#pragma unroll
  for (int j = 0; j < PER_LANE; ++j) acc[j] = 0.0f;

  for (int p = begin + warp; p < end; p += NWARP) {
    const size_t at = (size_t)p * kv_stride + kvh * HD + lane * PER_LANE;
    float kv[2][PER_LANE];
#pragma unroll
    for (int j = 0; j < PER_LANE; ++j) kv[0][j] = bf2f(k_cache[at + j]);
#pragma unroll
    for (int j = 0; j < PER_LANE; ++j) kv[1][j] = bf2f(v_cache[at + j]);

    float dot = 0.0f;
#pragma unroll
    for (int j = 0; j < PER_LANE; ++j) dot = fmaf(qr[j], kv[0][j], dot);
    dot = warp_all_reduce(dot) * scale;

    const float mn = fmaxf(m, dot);
    const float corr = __expf(m - mn);
    const float w = __expf(dot - mn);
#pragma unroll
    for (int j = 0; j < PER_LANE; ++j) acc[j] = acc[j] * corr + w * kv[1][j];
    s = s * corr + w;
    m = mn;
  }

  if (lane == 0) {
    red_m[warp] = m;
    red_s[warp] = s;
  }
#pragma unroll
  for (int j = 0; j < PER_LANE; ++j) red_acc[warp][lane * PER_LANE + j] = acc[j];
  __syncthreads();

  if (warp == 0) {
    float gm = -INFINITY;
#pragma unroll
    for (int t = 0; t < NWARP; ++t) gm = fmaxf(gm, red_m[t]);
    float gs = 0.0f;
    float o[PER_LANE];
#pragma unroll
    for (int j = 0; j < PER_LANE; ++j) o[j] = 0.0f;
#pragma unroll
    for (int t = 0; t < NWARP; ++t) {
      const float c = __expf(red_m[t] - gm);
      gs += red_s[t] * c;
#pragma unroll
      for (int j = 0; j < PER_LANE; ++j) o[j] += red_acc[t][lane * PER_LANE + j] * c;
    }
    const size_t base = ((size_t)h * SPLITS + split) * HD + lane * PER_LANE;
#pragma unroll
    for (int j = 0; j < PER_LANE; ++j) part_acc[base + j] = o[j];
    if (lane == 0) {
      part_ms[((size_t)h * SPLITS + split) * 2 + 0] = gm;
      part_ms[((size_t)h * SPLITS + split) * 2 + 1] = gs;
    }
  }
}

// Merge the position splits: one block per query head, rescaled against the
// joint maximum, then normalised and rounded once.
template <int HD, int SPLITS>
__global__ __launch_bounds__(HD) void attn_combine_kernel(
    const float *__restrict__ part_acc, const float *__restrict__ part_ms,
    __nv_bfloat16 *__restrict__ out) {
  const int h = blockIdx.x;
  const int d = threadIdx.x;
  float gm = -INFINITY;
#pragma unroll
  for (int t = 0; t < SPLITS; ++t) gm = fmaxf(gm, part_ms[((size_t)h * SPLITS + t) * 2]);
  float gs = 0.0f, o = 0.0f;
#pragma unroll
  for (int t = 0; t < SPLITS; ++t) {
    const float m = part_ms[((size_t)h * SPLITS + t) * 2 + 0];
    const float s = part_ms[((size_t)h * SPLITS + t) * 2 + 1];
    const float c = m == -INFINITY ? 0.0f : __expf(m - gm);
    gs += s * c;
    o += part_acc[((size_t)h * SPLITS + t) * HD + d] * c;
  }
  out[h * HD + d] = f2bf(gs > 0.0f ? o / gs : 0.0f);
}

// ---------------------------------------------------------------------------
// Mixture of experts
// ---------------------------------------------------------------------------

// Top-k over the router's logits and a softmax over just those k.
//
// The projection itself is an ordinary matvec, run before this: 72 rows over
// one block was the single worst kernel in a decode step, because one SM
// cannot pull 590 KB at any useful rate. What is left here is genuinely one
// block's worth of work -- 72 numbers, ten rounds of warp-wide argmax.
template <int PER_LANE>
__global__ __launch_bounds__(WARP) void topk_softmax_kernel(
    const float *__restrict__ logits, __nv_bfloat16 *__restrict__ weights,
    int64_t *__restrict__ indices, int num_experts, int topk) {
  const int lane = threadIdx.x;
  // Each lane keeps its own slice of the logits in registers and strikes out
  // its own winners, so a round is shuffles alone -- no shared memory, and no
  // barrier between rounds.
  float mine[PER_LANE];
  int which[PER_LANE];
#pragma unroll
  for (int j = 0; j < PER_LANE; ++j) {
    const int e = lane + j * WARP;
    which[j] = e;
    // Rounded to bf16 before the selection, not after: HF's router projection
    // lands at bf16 and only then widens, and which ten experts win is decided
    // on those rounded values. Keeping the f32 accumulator would be more
    // accurate and would sometimes pick a different expert, which is a
    // different model.
    mine[j] = e < num_experts ? bf2f(f2bf(logits[e])) : -INFINITY;
  }

  // Shared, not a local array: `topk` is a runtime value, so a local array
  // indexed by it would be spilled to local memory and read back per round.
  __shared__ float chosen[32];
  float best = 0.0f;
  for (int t = 0; t < topk; ++t) {
    float bv = -INFINITY;
    int bi = num_experts;
#pragma unroll
    for (int j = 0; j < PER_LANE; ++j) {
      if (mine[j] > bv || (mine[j] == bv && which[j] < bi)) {
        bv = mine[j];
        bi = which[j];
      }
    }
    // Largest first, ties to the lower index -- torch.topk's own order.
#pragma unroll
    for (int off = WARP / 2; off > 0; off >>= 1) {
      const float ov = __shfl_xor_sync(0xffffffffu, bv, off);
      const int oi = __shfl_xor_sync(0xffffffffu, bi, off);
      if (ov > bv || (ov == bv && oi < bi)) {
        bv = ov;
        bi = oi;
      }
    }
#pragma unroll
    for (int j = 0; j < PER_LANE; ++j)
      if (which[j] == bi) mine[j] = -INFINITY;
    if (lane == 0) indices[t] = bi;
    chosen[t] = bv;
    if (t == 0) best = bv;
  }

  if (lane == 0) {
    float total = 0.0f;
    for (int t = 0; t < topk; ++t) {
      chosen[t] = __expf(chosen[t] - best);
      total += chosen[t];
    }
    for (int t = 0; t < topk; ++t) weights[t] = f2bf(chosen[t] / total);
  }
}

// One chosen expert's fused input projection, both halves as one row sweep.
//
// Gate and up are the same shape from two tensors, so sweeping 2 * inter rows
// gives twice the rows in flight that computing the halves one after the other
// does -- and a matvec has nothing but rows in flight. The SwiGLU that joins
// them is a separate elementwise pass over 1/4096th of the bytes.
template <int ROWS, int NWARP, int UNROLL>
__global__ __launch_bounds__(NWARP *WARP) void experts_proj_kernel(
    const __nv_bfloat16 *__restrict__ w_gate, const __nv_bfloat16 *__restrict__ w_up,
    const __nv_bfloat16 *__restrict__ x, const int64_t *__restrict__ indices,
    __nv_bfloat16 *__restrict__ both, int inter, int n) {
  extern __shared__ uint4 xs[];
  stage_x(x, xs, n);

  const int slot = blockIdx.y;
  const int e = (int)indices[slot];
  const int lane = threadIdx.x & (WARP - 1);
  const int warp = threadIdx.x >> 5;
  const int row0 = (blockIdx.x * NWARP + warp) * ROWS;
  if (row0 >= 2 * inter) return;
  const int live = min(ROWS, 2 * inter - row0);

  const __nv_bfloat16 *half = row0 < inter ? w_gate : w_up;
  const int local = row0 < inter ? row0 : row0 - inter;
  float acc[ROWS];
  rows_dot<ROWS, UNROLL>(half + (size_t)e * inter * n + (size_t)local * n, n, n, xs, lane, live,
                         acc);

  if (lane == 0) {
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      if (r >= live) break;
      both[(size_t)slot * 2 * inter + row0 + r] = f2bf(acc[r]);
    }
  }
}

// One chosen expert's down projection, already scaled by its routing weight.
//
// The ten experts are a grid axis rather than a loop, which is what gets this
// shape off the ground: 768 is a short reduction, so a lane cannot hold many
// loads along it, and the parallelism has to come from somewhere. Ten times as
// many blocks is that somewhere; `experts_mix_kernel` adds the partials up.
template <int ROWS, int NWARP, int UNROLL>
__global__ __launch_bounds__(NWARP *WARP) void experts_down_kernel(
    const __nv_bfloat16 *__restrict__ w_down, const __nv_bfloat16 *__restrict__ h,
    const int64_t *__restrict__ indices, const __nv_bfloat16 *__restrict__ weights,
    float *__restrict__ partial, int hidden, int inter) {
  extern __shared__ uint4 hs[];
  const int slot = blockIdx.y;
  stage_x(h + (size_t)slot * inter, hs, inter);

  const int e = (int)indices[slot];
  const float g = bf2f(weights[slot]);
  const int lane = threadIdx.x & (WARP - 1);
  const int warp = threadIdx.x >> 5;
  const int row0 = (blockIdx.x * NWARP + warp) * ROWS;
  if (row0 >= hidden) return;
  const int live = min(ROWS, hidden - row0);

  float acc[ROWS];
  rows_dot<ROWS, UNROLL>(w_down + (size_t)e * hidden * inter + (size_t)row0 * inter, inter,
                         inter, hs, lane, live, acc);

  if (lane == 0) {
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      if (r >= live) break;
      partial[(size_t)slot * hidden + row0 + r] = g * bf2f(f2bf(acc[r]));
    }
  }
}

// The routed block: every chosen expert's contribution, summed in f32 and
// landed at bf16 once -- where the reference's own reduction lands it.
__global__ void experts_mix_kernel(const float *__restrict__ partial,
                                   __nv_bfloat16 *__restrict__ out, int hidden, int topk) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= hidden) return;
  float total = 0.0f;
  for (int t = 0; t < topk; ++t) total += partial[(size_t)t * hidden + i];
  out[i] = f2bf(total);
}

// SwiGLU over one fused projection: the first half activated, the second the
// multiplicand. Splitting this out of the projection lets the projection be an
// ordinary matvec over all 3072 rows at once, which is twice the rows in flight
// that computing the two halves separately would give.
__global__ void swiglu_kernel(const __nv_bfloat16 *__restrict__ both,
                              __nv_bfloat16 *__restrict__ h, int inter, int batch) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= inter * batch) return;
  const int slot = i / inter, off = i - slot * inter;
  const __nv_bfloat16 *row = both + (size_t)slot * 2 * inter;
  h[i] = f2bf(silu_f(bf2f(row[off])) * bf2f(row[inter + off]));
}

// ---------------------------------------------------------------------------
// Embedding and sampling
// ---------------------------------------------------------------------------

// The token id lives on the device, so a captured step reads whatever the
// previous step's sampler wrote there without a host round trip.
__global__ void embed_kernel(const __nv_bfloat16 *__restrict__ table,
                             const int64_t *__restrict__ token_id,
                             __nv_bfloat16 *__restrict__ out, int hidden, float mult) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= hidden) return;
  const int64_t tok = token_id[0];
  out[i] = f2bf(bf2f(table[tok * (size_t)hidden + i]) * mult);
}

// Argmax and categorical sampling share one shape: score every logit, keep the
// best. Greedy scores the logit itself; sampling adds a Gumbel variate, whose
// argmax is exactly a draw from softmax(logits) -- no normalisation pass, and
// one kernel either way.
__global__ void sample_stage1_kernel(const float *__restrict__ logits, int vocab,
                                     uint64_t seed, const int *__restrict__ step, int greedy,
                                     float temperature, float *__restrict__ part_val,
                                     int32_t *__restrict__ part_idx) {
  __shared__ float sv[256];
  __shared__ int si[256];

  float best = -INFINITY;
  int bi = 0;
  const int stride = gridDim.x * blockDim.x;
  const int start = blockIdx.x * blockDim.x + threadIdx.x;

  curandStatePhilox4_32_10_t st;
  if (!greedy) curand_init(seed, (uint64_t)start, (uint64_t)(*step) * 4ull, &st);

  for (int i = start; i < vocab; i += stride) {
    float v = logits[i];
    if (!greedy) {
      v /= temperature;
      // curand_uniform is (0, 1]; at exactly 1 the Gumbel variate would be
      // +inf and every masked logit would tie for the win, so the top end is
      // pulled in by one ulp before the double log.
      const float u = fminf(curand_uniform(&st), 0.99999994f);
      v += -__logf(-__logf(u));
    }
    if (v > best) {
      best = v;
      bi = i;
    }
  }
  sv[threadIdx.x] = best;
  si[threadIdx.x] = bi;
  __syncthreads();
  for (int off = blockDim.x >> 1; off > 0; off >>= 1) {
    if (threadIdx.x < off) {
      if (sv[threadIdx.x + off] > sv[threadIdx.x]) {
        sv[threadIdx.x] = sv[threadIdx.x + off];
        si[threadIdx.x] = si[threadIdx.x + off];
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    part_val[blockIdx.x] = sv[0];
    part_idx[blockIdx.x] = si[0];
  }
}

// The winner of the partials becomes the next step's input token and the entry
// this step contributes to the transcript, and the write cursor moves on. All
// three live on the device, so one captured graph is the whole decode step.
__global__ void sample_stage2_kernel(const float *__restrict__ part_val,
                                     const int32_t *__restrict__ part_idx, int parts,
                                     int64_t *__restrict__ token_id,
                                     int32_t *__restrict__ sampled, int *__restrict__ pos,
                                     int advance) {
  __shared__ float sv[64];
  __shared__ int si[64];
  float best = -INFINITY;
  int bi = 0;
  for (int i = threadIdx.x; i < parts; i += blockDim.x) {
    if (part_val[i] > best) {
      best = part_val[i];
      bi = part_idx[i];
    }
  }
  sv[threadIdx.x] = best;
  si[threadIdx.x] = bi;
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int i = 1; i < blockDim.x; ++i) {
      if (sv[i] > sv[0]) {
        sv[0] = sv[i];
        si[0] = si[i];
      }
    }
    const int p = *pos;
    sampled[p] = si[0];
    token_id[0] = si[0];
    if (advance) *pos = p + 1;
  }
}

}  // namespace

// ===========================================================================
// Host surface
//
// Every entry takes its output tensor from the caller. Nothing here allocates
// and nothing synchronises, so the whole step captures into one CUDA graph and
// replays as a single launch -- which matters when a step is four hundred
// kernels and each one is a few microseconds of work.
// ===========================================================================

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

namespace {

// One row per warp, sixteen warps per block. Swept over every projection shape
// in the model (`tools/tune_gemv.py`): a matvec wants warps resident per SM
// more than it wants rows batched per warp, because there is no reuse for the
// batching to pay for -- the rows only ever share the staged x vector, and that
// is already in shared memory.
constexpr int GEMV_ROWS = 1;
constexpr int GEMV_WARPS = 16;
constexpr int GEMV_TILE = GEMV_ROWS * GEMV_WARPS;

inline cudaStream_t stream() { return at::cuda::getCurrentCUDAStream(); }

inline const __nv_bfloat16 *bf(const torch::Tensor &t) {
  return reinterpret_cast<const __nv_bfloat16 *>(t.data_ptr<at::BFloat16>());
}
inline __nv_bfloat16 *bfw(torch::Tensor &t) {
  return reinterpret_cast<__nv_bfloat16 *>(t.data_ptr<at::BFloat16>());
}

inline int tiles(int m) { return (m + GEMV_TILE - 1) / GEMV_TILE; }

}  // namespace

// Launch shapes. ROWS * UNROLL loads in flight per lane is what these encode;
// the reduction length decides how much unrolling a lane can actually reach.
namespace {

template <int UNROLL, int FP32_OUT>
void launch_gemv(const torch::Tensor &w, const torch::Tensor &x, void *y, float alpha) {
  const int m = (int)w.size(0), n = (int)w.size(-1);
  gemv_kernel<GEMV_ROWS, GEMV_WARPS, UNROLL, FP32_OUT>
      <<<tiles(m), GEMV_WARPS * WARP, n * 2, stream()>>>(bf(w), bf(x), y, m, n, alpha);
}

//: The longest unroll a reduction of *n* can fill at least once.
inline int unroll_for(int n) { return (n >> 3) >= 4 * WARP ? 4 : ((n >> 3) >= 2 * WARP ? 2 : 1); }

//: Shared memory for a matvec block: the staged input vector, nothing else.
inline int gemv_smem(int n) { return n * 2; }

}  // namespace

// A tuning hook: the same kernel with its shape knobs opened up, so
// `tools/tune_gemv.py` can measure a configuration rather than argue about it.
// Not on the decode path -- `gemv` below carries the answers it found.
template <int ROWS, int NWARP, int UNROLL>
static void launch_tuned(const torch::Tensor &w, const torch::Tensor &x, torch::Tensor y) {
  const int m = (int)w.size(0), n = (int)w.size(-1);
  const int per_block = ROWS * NWARP;
  gemv_kernel<ROWS, NWARP, UNROLL, 0>
      <<<(m + per_block - 1) / per_block, NWARP * WARP, n * 2, stream()>>>(
          bf(w), bf(x), bfw(y), m, n, 0.0f);
}

void gemv_tuned(const torch::Tensor &w, const torch::Tensor &x, torch::Tensor y, int64_t rows,
                int64_t warps, int64_t unroll) {
#define TF_CASE(R, W, U)                                    \
  if (rows == (R) && warps == (W) && unroll == (U)) {       \
    launch_tuned<R, W, U>(w, x, y);                         \
    return;                                                 \
  }
#define TF_ROW(R)                                           \
  TF_CASE(R, 2, 1) TF_CASE(R, 2, 2) TF_CASE(R, 2, 4)        \
  TF_CASE(R, 4, 1) TF_CASE(R, 4, 2) TF_CASE(R, 4, 4)        \
  TF_CASE(R, 8, 1) TF_CASE(R, 8, 2) TF_CASE(R, 8, 4)        \
  TF_CASE(R, 16, 1) TF_CASE(R, 16, 2) TF_CASE(R, 16, 4)
  TF_ROW(1) TF_ROW(2) TF_ROW(4) TF_ROW(8)
#undef TF_ROW
#undef TF_CASE
  TORCH_CHECK(false, "gemv_tuned: no instantiation for that shape");
}

void gemv(const torch::Tensor &w, const torch::Tensor &x, torch::Tensor y) {
  switch (unroll_for((int)w.size(-1))) {
    case 4: launch_gemv<4, 0>(w, x, bfw(y), 0.0f); break;
    case 2: launch_gemv<2, 0>(w, x, bfw(y), 0.0f); break;
    default: launch_gemv<1, 0>(w, x, bfw(y), 0.0f); break;
  }
}

void gemv_f32(const torch::Tensor &w, const torch::Tensor &x, torch::Tensor y, double alpha) {
  switch (unroll_for((int)w.size(-1))) {
    case 4: launch_gemv<4, 1>(w, x, y.data_ptr<float>(), (float)alpha); break;
    case 2: launch_gemv<2, 1>(w, x, y.data_ptr<float>(), (float)alpha); break;
    default: launch_gemv<1, 1>(w, x, y.data_ptr<float>(), (float)alpha); break;
  }
}

void residual_add(const torch::Tensor &a, const torch::Tensor &b, torch::Tensor out,
                  double mult) {
  const int n = (int)a.numel();
  residual_add_kernel<<<(n + 255) / 256, 256, 0, stream()>>>(bf(a), bf(b), bfw(out), n,
                                                             (float)mult);
}

void cache_write(torch::Tensor cache, const torch::Tensor &entry,
                 const torch::Tensor &cur_pos) {
  const int width = (int)entry.numel();
  cache_write_kernel<<<(width + 255) / 256, 256, 0, stream()>>>(bfw(cache), bf(entry),
                                                                cur_pos.data_ptr<int>(), width);
}

void rms_norm(const torch::Tensor &x, const torch::Tensor &w, torch::Tensor y, double eps) {
  const int n = (int)x.numel();
  rms_norm_kernel<512><<<1, 512, 0, stream()>>>(bf(x), bf(w), bfw(y), n, (float)eps);
}

void rms_norm_gated(const torch::Tensor &x, const torch::Tensor &gate, const torch::Tensor &w,
                    torch::Tensor y, double eps) {
  const int n = (int)x.numel();
  rms_norm_gated_kernel<512><<<1, 512, n * 4, stream()>>>(bf(x), bf(gate), bf(w), bfw(y), n,
                                                          (float)eps);
}

void mamba_conv(torch::Tensor state, const torch::Tensor &col, const torch::Tensor &w,
                const torch::Tensor &b, torch::Tensor out) {
  const int channels = (int)w.size(0), krn = (int)w.size(1);
  const int win = (int)state.size(-1);
  mamba_conv_kernel<<<(channels + 255) / 256, 256, 0, stream()>>>(
      bfw(state), bf(col), bf(w), bf(b), bfw(out), channels, win, krn);
}

void mamba_ssm(torch::Tensor ssm, const torch::Tensor &x, const torch::Tensor &b_vec,
               const torch::Tensor &c_vec, const torch::Tensor &dt_raw,
               const torch::Tensor &a_log, const torch::Tensor &dt_bias,
               const torch::Tensor &d_skip, torch::Tensor y) {
  const int nh = (int)ssm.size(-3), pd = (int)ssm.size(-2), ns = (int)ssm.size(-1);
  constexpr int NW = 8;
  const int pairs = nh * pd;
  mamba_ssm_kernel<NW><<<(pairs + NW - 1) / NW, NW * WARP, 2 * ns * sizeof(float), stream()>>>(
      ssm.data_ptr<float>(), bf(x), bf(b_vec), bf(c_vec), bf(dt_raw), bf(a_log), bf(dt_bias),
      bf(d_skip), bfw(y), nh, pd, ns);
}

//: Position stripes per query head. Eight turns 32 blocks into 256, which is
//: roughly the machine, and holds the workspace at 32 x 8 x 128 floats.
constexpr int ATTN_SPLITS = 8;

void attn_decode(const torch::Tensor &q, const torch::Tensor &k_cache,
                 const torch::Tensor &v_cache, const torch::Tensor &cur_pos,
                 torch::Tensor part_acc, torch::Tensor part_ms, torch::Tensor out,
                 int64_t hq, int64_t hkv, double scale) {
  attn_decode_kernel<8, 128, ATTN_SPLITS><<<dim3((int)hq, ATTN_SPLITS), 8 * WARP, 0, stream()>>>(
      bf(q), bf(k_cache), bf(v_cache), cur_pos.data_ptr<int>(), part_acc.data_ptr<float>(),
      part_ms.data_ptr<float>(), (int)hkv, (int)(hq / hkv), (float)scale);
  attn_combine_kernel<128, ATTN_SPLITS><<<(int)hq, 128, 0, stream()>>>(
      part_acc.data_ptr<float>(), part_ms.data_ptr<float>(), bfw(out));
}

void topk_softmax(const torch::Tensor &logits, torch::Tensor weights, torch::Tensor indices) {
  // 72 experts over 32 lanes: three slots each, one spare.
  topk_softmax_kernel<4><<<1, WARP, 0, stream()>>>(
      logits.data_ptr<float>(), bfw(weights), indices.data_ptr<int64_t>(),
      (int)logits.numel(), (int)weights.numel());
}

// Tuning hooks for the two MoE matvecs, measured by `tools/tune_experts.py`.
// Not on the decode path; the shipped launches below carry what it found.
void experts_proj_tuned(const torch::Tensor &w_gate, const torch::Tensor &w_up,
                        const torch::Tensor &x, const torch::Tensor &indices,
                        torch::Tensor both, int64_t rows, int64_t warps, int64_t unroll) {
  const int inter = (int)w_gate.size(1), n = (int)w_gate.size(2);
  const int topk = (int)indices.numel();
#define TF_P(R, W, U)                                                                     \
  if (rows == (R) && warps == (W) && unroll == (U)) {                                     \
    experts_proj_kernel<R, W, U><<<dim3((2 * inter + (R) * (W) - 1) / ((R) * (W)), topk), \
                                   (W)*WARP, gemv_smem(n), stream()>>>(                   \
        bf(w_gate), bf(w_up), bf(x), indices.data_ptr<int64_t>(), bfw(both), inter, n);    \
    return;                                                                                \
  }
#define TF_PR(R) TF_P(R, 4, 2) TF_P(R, 4, 4) TF_P(R, 8, 2) TF_P(R, 8, 4) TF_P(R, 16, 2) TF_P(R, 16, 4)
  TF_PR(1) TF_PR(2) TF_PR(4) TF_PR(8)
#undef TF_PR
#undef TF_P
  TORCH_CHECK(false, "experts_proj_tuned: no instantiation");
}

void experts_down_tuned(const torch::Tensor &w_down, const torch::Tensor &h,
                        const torch::Tensor &indices, const torch::Tensor &weights,
                        torch::Tensor partial, int64_t rows, int64_t warps, int64_t unroll) {
  const int hidden = (int)w_down.size(1), inter = (int)w_down.size(2);
  const int topk = (int)indices.numel();
#define TF_D(R, W, U)                                                                  \
  if (rows == (R) && warps == (W) && unroll == (U)) {                                  \
    experts_down_kernel<R, W, U><<<dim3((hidden + (R) * (W)-1) / ((R) * (W)), topk),   \
                                   (W)*WARP, inter * 2, stream()>>>(                   \
        bf(w_down), bf(h), indices.data_ptr<int64_t>(), bf(weights),                   \
        partial.data_ptr<float>(), hidden, inter);                                      \
    return;                                                                             \
  }
#define TF_DR(R) TF_D(R, 4, 1) TF_D(R, 4, 2) TF_D(R, 8, 1) TF_D(R, 8, 2) TF_D(R, 16, 1) TF_D(R, 16, 2)
  TF_DR(1) TF_DR(2) TF_DR(4) TF_DR(8)
#undef TF_DR
#undef TF_D
  TORCH_CHECK(false, "experts_down_tuned: no instantiation");
}

void experts_gate_up(const torch::Tensor &w_gate, const torch::Tensor &w_up,
                     const torch::Tensor &x, const torch::Tensor &indices,
                     torch::Tensor both, torch::Tensor h) {
  const int inter = (int)w_gate.size(1), n = (int)w_gate.size(2);
  const int topk = (int)indices.numel();
  // Four rows per warp over eight warps, swept by `tools/tune_experts.py`. The
  // expert projection prefers rows to warps where the dense ones prefer the
  // opposite -- its ten slabs are scattered, so a block that reads more of one
  // of them at a time does better.
  constexpr int PR = 4, PW = 8;
  dim3 grid((2 * inter + PR * PW - 1) / (PR * PW), topk);
  experts_proj_kernel<PR, PW, 2><<<grid, PW * WARP, gemv_smem(n), stream()>>>(
      bf(w_gate), bf(w_up), bf(x), indices.data_ptr<int64_t>(), bfw(both), inter, n);
  const int elems = topk * inter;
  swiglu_kernel<<<(elems + 255) / 256, 256, 0, stream()>>>(bf(both), bfw(h), inter, topk);
}

void experts_down(const torch::Tensor &w_down, const torch::Tensor &h,
                  const torch::Tensor &indices, const torch::Tensor &weights,
                  torch::Tensor partial, torch::Tensor out) {
  const int hidden = (int)w_down.size(1), inter = (int)w_down.size(2);
  const int topk = (int)indices.numel();
  // Four rows per warp and only four warps: the reduction is 768 long, so the
  // rows are where a lane's loads in flight have to come from, and a smaller
  // block leaves room for more of them per SM.
  constexpr int DR = 4, DW = 4;
  experts_down_kernel<DR, DW, 1>
      <<<dim3((hidden + DR * DW - 1) / (DR * DW), topk), DW * WARP, inter * 2, stream()>>>(
          bf(w_down), bf(h), indices.data_ptr<int64_t>(), bf(weights),
          partial.data_ptr<float>(), hidden, inter);
  experts_mix_kernel<<<(hidden + 255) / 256, 256, 0, stream()>>>(
      partial.data_ptr<float>(), bfw(out), hidden, topk);
}

void swiglu(const torch::Tensor &both, torch::Tensor h) {
  const int inter = (int)h.numel();
  swiglu_kernel<<<(inter + 255) / 256, 256, 0, stream()>>>(bf(both), bfw(h), inter, 1);
}

void embed(const torch::Tensor &table, const torch::Tensor &token_id, torch::Tensor out,
           double mult) {
  const int hidden = (int)table.size(1);
  embed_kernel<<<(hidden + 255) / 256, 256, 0, stream()>>>(
      bf(table), token_id.data_ptr<int64_t>(), bfw(out), hidden, (float)mult);
}

void sample(const torch::Tensor &logits, torch::Tensor part_val, torch::Tensor part_idx,
            torch::Tensor token_id, torch::Tensor sampled, torch::Tensor pos, int64_t seed,
            bool greedy, double temperature, bool advance) {
  const int vocab = (int)logits.numel();
  const int parts = (int)part_val.numel();
  sample_stage1_kernel<<<parts, 256, 0, stream()>>>(
      logits.data_ptr<float>(), vocab, (uint64_t)seed, pos.data_ptr<int>(), greedy ? 1 : 0,
      (float)temperature, part_val.data_ptr<float>(), part_idx.data_ptr<int32_t>());
  sample_stage2_kernel<<<1, 64, 0, stream()>>>(
      part_val.data_ptr<float>(), part_idx.data_ptr<int32_t>(), parts,
      token_id.data_ptr<int64_t>(), sampled.data_ptr<int32_t>(), pos.data_ptr<int>(),
      advance ? 1 : 0);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gemv", &gemv);
  m.def("gemv_tuned", &gemv_tuned);
  m.def("gemv_f32", &gemv_f32);
  m.def("residual_add", &residual_add);
  m.def("cache_write", &cache_write);
  m.def("rms_norm", &rms_norm);
  m.def("rms_norm_gated", &rms_norm_gated);
  m.def("mamba_conv", &mamba_conv);
  m.def("mamba_ssm", &mamba_ssm);
  m.def("attn_decode", &attn_decode);
  m.def("topk_softmax", &topk_softmax);
  m.def("experts_gate_up", &experts_gate_up);
  m.def("experts_proj_tuned", &experts_proj_tuned);
  m.def("experts_down_tuned", &experts_down_tuned);
  m.def("experts_down", &experts_down);
  m.def("swiglu", &swiglu);
  m.def("embed", &embed);
  m.def("sample", &sample);
}
