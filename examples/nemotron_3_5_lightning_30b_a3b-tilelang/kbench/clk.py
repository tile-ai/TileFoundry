import tilelang, tilelang.language as T, torch
@tilelang.jit
def build():
    @T.prim_func
    def main(out: T.Tensor((4,), "int64")):
        with T.Kernel(1, threads=32) as bx:
            tid = T.get_thread_binding()
            if tid == 0:
                out[0] = T.call_extern("int64", "clock64")
                for _i in T.serial(10000):
                    out[2] = out[2] + 1
                out[1] = T.call_extern("int64", "clock64")
    return main
o = torch.zeros(4, dtype=torch.int64, device="cuda")
build()(o); torch.cuda.synchronize()
print(o.tolist(), "delta", (o[1] - o[0]).item())
