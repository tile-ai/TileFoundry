"""Authored HIR sketches for FlashInfer attention and collectives.

Notes:
upstream: flashinfer-ai/flashinfer @ 2ab910c58fdd2392914ea05e2a8714946ac0eef6
license: Apache-2.0 (no upstream source is vendored)
blocked: refused
phase: selection/analysis
error: source defines no TileFoundry Module
classification: expected-spec notation; no authored Module is declared.
This expected-spec corpus is intentionally not required to parse.
"""

from tilefoundry.dsl.tf import *
from tilefoundry.ir.types.shard import Layout, Mesh, Topology

CTA = Mesh((Topology("cta", 1),), Layout(shape=(1,), strides=(1,)))
THREADS = Mesh((Topology("thread", 128),), Layout(shape=(128,), strides=(1,)))
GPU = Mesh((Topology("gpu", 8),), Layout(shape=(8,), strides=(1,)))


def _place(x, scope, storage, tile=None):

    return place(x, scope=scope, storage=storage, tile=tile)


def _cta(x, tile=None):
    return _place(x, CTA, "smem", tile)


def _thread(x, tile=None):
    return _place(x, THREADS, "rmem", tile)


def _gpu(x, tile=None):
    return _place(x, GPU, "gmem", tile)


# noqa
def A01_prefill(q, k, v, mask):
    return normalize(online_softmax(where(mask, matmul(q, transpose(k, (-1, -2))), -inf), value=v))


def A01_prefill_cta(q, k, v, mask):
    return _cta(A01_prefill(q, k, v, mask), (64, 64, 64))


def A01_prefill_thread(q, k, v, mask):
    return _thread(A01_prefill(q, k, v, mask), (16, 16, 16))


# noqa
def A02_paged_decode(q, pk, pv, indptr, indices):
    return A01_prefill(
        q, paged_gather(pk, indptr, indices), paged_gather(pv, indptr, indices), None
    )


def A02_paged_decode_cta(q, pk, pv, indptr, indices):
    return _cta(A02_paged_decode(q, pk, pv, indptr, indices), (1, 64, 64))


def A02_paged_decode_thread(q, pk, pv, indptr, indices):
    return _thread(A02_paged_decode(q, pk, pv, indptr, indices), (1, 16, 16))


# noqa
def A03_softcap(q, k, v):
    s = matmul(q, transpose(k, (-1, -2))) * scale
    return normalize(online_softmax(cap * tanh(s / cap), value=v))


def A03_softcap_cta(q, k, v):
    return _cta(A03_softcap(q, k, v), (64, 64, 64))


def A03_softcap_thread(q, k, v):
    return _thread(A03_softcap(q, k, v), (16, 16, 16))


# noqa
def A04_alibi(q, k, v, slope):
    return normalize(
        online_softmax(
            matmul(q, transpose(k, (-1, -2))) * scale + slope * (kv_position - q_position), value=v
        )
    )


def A04_alibi_cta(q, k, v, slope):
    return _cta(A04_alibi(q, k, v, slope), (64, 64, 64))


def A04_alibi_thread(q, k, v, slope):
    return _thread(A04_alibi(q, k, v, slope), (16, 16, 16))


# noqa
def A05_masked(q, k, v, bits):
    return normalize(
        online_softmax(masked(matmul(q, transpose(k, (-1, -2))) * scale, bits), value=v)
    )


def A05_masked_cta(q, k, v, bits):
    return _cta(A05_masked(q, k, v, bits), (64, 64, 64))


def A05_masked_thread(q, k, v, bits):
    return _thread(A05_masked(q, k, v, bits), (16, 16, 16))


# noqa
def A06_pod(q, k, v, mode, indptr=None, indices=None):
    return dynamic_select(
        mode, A01_prefill(q, k, v, None), A02_paged_decode(q, k, v, indptr, indices)
    )


def A06_pod_cta(q, k, v, mode):
    return _cta(A06_pod(q, k, v, mode), (64, 64, 64))


def A06_pod_thread(q, k, v, mode):
    return _thread(A06_pod(q, k, v, mode), (16, 16, 16))


# noqa
def A07_merge(o0, m0, d0, o1, m1, d1):
    m = maximum(m0, m1)
    d = exp(m0 - m) * d0 + exp(m1 - m) * d1
    return (exp(m0 - m) * o0 + exp(m1 - m) * o1) / d, m, d


def A07_merge_cta(*xs):
    return _cta(A07_merge(*xs))


def A07_merge_thread(*xs):
    return _thread(A07_merge(*xs))


# noqa
def C01(x, r):
    return rms_norm(all_reduce(x) + r)


def C01_cta(x, r):
    return _cta(C01(x, r))


def C01_thread(x, r):
    return _thread(C01(x, r))


def C01_gpu(x, r):
    return _gpu(C01(x, r))


def C02(x, r):
    return block_quant(rms_norm(all_reduce(x) + r), format="fp8")


def C02_cta(x, r):
    return _cta(C02(x, r))


def C02_thread(x, r):
    return _thread(C02(x, r))


def C02_gpu(x, r):
    return _gpu(C02(x, r))


def C03(x):
    return rms_norm(all_reduce(x))


def C03_cta(x):
    return _cta(C03(x))


def C03_thread(x):
    return _thread(C03(x))


def C03_gpu(x):
    return _gpu(C03(x))


def C04(expert, r):
    return block_quant(rms_norm(all_reduce(expert_reduce(expert)) + r), format="fp8")


def C04_cta(expert, r):
    return _cta(C04(expert, r))


def C04_thread(expert, r):
    return _thread(C04(expert, r))


def C04_gpu(expert, r):
    return _gpu(C04(expert, r))


def C05(permuted, shared):
    return rms_norm(all_reduce(expert_finalize(permuted) + shared))


def C05_cta(permuted, shared):
    return _cta(C05(permuted, shared))


def C05_thread(permuted, shared):
    return _thread(C05(permuted, shared))


def C05_gpu(permuted, shared):
    return _gpu(C05(permuted, shared))


# noqa
def C06(x, w):
    return matmul(all_gather_tiles(x), w)


def C06_cta(x, w):
    return _cta(C06(x, w), (128, 64, 32))


def C06_thread(x, w):
    return _thread(C06(x, w), (16, 16, 16))


def C06_gpu(x, w):
    return _gpu(C06(x, w))


# noqa
def C07(a, b):
    return two_shot_all_reduce(matmul(a, b), phases=("reduce_scatter", "all_gather"))


def C07_cta(a, b):
    return _cta(C07(a, b), (128, 128, 32))


def C07_thread(a, b):
    return _thread(C07(a, b), (16, 16, 16))


def C07_gpu(a, b):
    return _gpu(C07(a, b))
