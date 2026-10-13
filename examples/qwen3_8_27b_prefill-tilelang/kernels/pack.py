"""Weight packing kernels: the TIR of scheduled/gemm.py's pack funcs.

Each reads only checkpoint weights, so runtime_model.py runs it once per layer and
holds the result. 132 CTAs; CTA c copies 128-row blocks c, c + 132, ... of
each source into its place in the packed weight (gmem to gmem).

  gate_up  [w_gate rows 128 j .. | w_up rows 128 j ..] per 256-row tile
  in_proj  [w_qkv | w_z | w_b | w_a | 160 zero rows]  = 16640 rows
  qkv      [w_q | w_k | w_v]                           = 14336 rows
"""
import tilelang
import tilelang.language as T

BF16 = "bfloat16"
NUM_SM, H, I = 132, 5120, 17408


def gate_up_kernel():
    @T.prim_func
    def pack(Wg: T.Tensor((I, H), BF16), Wu: T.Tensor((I, H), BF16), W: T.Tensor((2 * I, H), BF16)):
        with T.Kernel(NUM_SM, threads=384) as c:
            for j in T.serial(T.ceildiv(I // 128 - c, NUM_SM)):
                b = c + j * NUM_SM
                T.copy(Wg[b * 128:(b + 1) * 128, :], W[b * 256:b * 256 + 128, :])
                T.copy(Wu[b * 128:(b + 1) * 128, :], W[b * 256 + 128:(b + 1) * 256, :])
    return pack


def in_proj_kernel():
    @T.prim_func
    def pack(Wqkv: T.Tensor((10240, H), BF16), Wz: T.Tensor((6144, H), BF16), Wb: T.Tensor((48, H), BF16),
             Wa: T.Tensor((48, H), BF16), W: T.Tensor((16640, H), BF16)):
        with T.Kernel(NUM_SM, threads=384) as c:
            for j in T.serial(T.ceildiv(80 - c, NUM_SM)):
                b = c + j * NUM_SM
                T.copy(Wqkv[b * 128:(b + 1) * 128, :], W[b * 128:(b + 1) * 128, :])
            for j in T.serial(T.ceildiv(48 - c, NUM_SM)):
                b = c + j * NUM_SM
                T.copy(Wz[b * 128:(b + 1) * 128, :], W[10240 + b * 128:10240 + (b + 1) * 128, :])
            if c == 0:
                T.copy(Wb, W[16384:16432, :])
                T.copy(Wa, W[16432:16480, :])
                for i, k in T.Parallel(160, H):
                    W[16480 + i, k] = T.cast(0, BF16)
    return pack


def qkv_kernel():
    @T.prim_func
    def pack(Wq: T.Tensor((12288, H), BF16), Wk: T.Tensor((1024, H), BF16), Wv: T.Tensor((1024, H), BF16),
             W: T.Tensor((14336, H), BF16)):
        with T.Kernel(NUM_SM, threads=384) as c:
            for j in T.serial(T.ceildiv(96 - c, NUM_SM)):
                b = c + j * NUM_SM
                T.copy(Wq[b * 128:(b + 1) * 128, :], W[b * 128:(b + 1) * 128, :])
            for j in T.serial(T.ceildiv(8 - c, NUM_SM)):
                b = c + j * NUM_SM
                T.copy(Wk[b * 128:(b + 1) * 128, :], W[12288 + b * 128:12288 + (b + 1) * 128, :])
                T.copy(Wv[b * 128:(b + 1) * 128, :], W[13312 + b * 128:13312 + (b + 1) * 128, :])
    return pack


_CACHE = {}


def compiled(name):
    if name not in _CACHE:
        _CACHE[name] = tilelang.compile({"gate_up": gate_up_kernel, "in_proj": in_proj_kernel, "qkv": qkv_kernel}[name](),
                                        out_idx=None)
    return _CACHE[name]
