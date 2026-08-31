import tilelang, tilelang.language as T, torch
@tilelang.jit
def build():
    @T.prim_func
    def main(sc: T.Tensor((128,), "float32"), out: T.Tensor((8,), "float32")):
        with T.Kernel(1, threads=256) as bx:
            fs = T.alloc_shared((256,), "float32")
            sel = T.alloc_shared((16,), "float32")
            fv = T.alloc_local((4,), "float32")
            iv = T.alloc_local((4,), "int32")
            tid = T.get_thread_binding()
            lane = tid % 32
            if tid < 128:
                fs[tid] = sc[tid]
            T.sync_threads()
            if tid < 32:
                for _j in T.serial(6):
                    fv[0] = -1e30
                    iv[0] = 0
                    for _q in T.serial(4):
                        if fs[_q * 32 + lane] > fv[0]:
                            fv[0] = fs[_q * 32 + lane]
                            iv[0] = _q * 32 + lane
                    for _d in T.unroll(5):
                        fv[1] = T.shfl_xor(fv[0], T.shift_left(1, _d))
                        iv[1] = T.shfl_xor(iv[0], T.shift_left(1, _d))
                        if fv[1] > fv[0]:
                            fv[0] = fv[1]
                            iv[0] = iv[1]
                        else:
                            if fv[1] == fv[0]:
                                if iv[1] < iv[0]:
                                    iv[0] = iv[1]
                    if lane == 0:
                        sel[_j] = T.Cast("float32", iv[0])
                        sel[6 + _j] = fv[0]
                        fs[iv[0]] = -1e30
                    T.sync_warp()
            T.sync_threads()
            if tid < 6:
                out[tid] = sel[tid]
            if tid == 7:
                out[7] = sel[6]
    return main
sc = torch.randn(128, device="cuda")
o = torch.zeros(8, device="cuda")
build()(sc, o); torch.cuda.synchronize()
v, i = sc.topk(6)
print("kernel", [int(x) for x in o[:6].tolist()], "top1val", o[7].item())
print("torch ", i.tolist(), "top1val", v[0].item())
