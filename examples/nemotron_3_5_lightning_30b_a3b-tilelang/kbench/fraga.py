"""Can the query side of the scores product live in registers?

Q is 32 heads x 128 dims and never changes inside a step, so 8 KB of shared
memory holds it for the whole attention stage -- against a shared budget that is
already 200 bytes from the edge.  In registers it is four per thread.
"""
import torch
import tilelang
import tilelang.language as T

GQA, DH, ABLK, KVROW, THREADS = 16, 128, 128, 256, 256


@tilelang.jit
def build(where):
    @T.prim_func
    def main(q: T.Tensor((GQA, DH), "float32"), k: T.Tensor((ABLK, KVROW), "bfloat16"),
             o: T.Tensor((GQA, ABLK), "float32")):
        with T.Kernel(1, threads=THREADS) as bx:
            ks = T.alloc_shared((ABLK, KVROW), "bfloat16")
            f32 = T.alloc_shared((GQA, DH), "float32")
            qf = T.alloc_fragment((GQA, DH), "bfloat16")
            qs = T.alloc_shared((GQA, DH), "bfloat16")
            sf = T.alloc_fragment((GQA, ABLK), "float32")
            of = T.alloc_fragment((GQA, DH), "float32")
            T.copy(k, ks)
            T.copy(q, f32)
            T.clear(sf)
            T.clear(of)
            if where == "fragment":
                # f32 shared -> bf16 fragment, then straight into the gemm
                T.copy(f32, qf)
                T.gemm(qf, ks[:, 0:DH], sf, transpose_B=True)
            elif where == "pv":
                T.copy(f32, qf)
                T.gemm(qf, ks[0:DH, 0:DH], of)
                T.copy(of, sf[:, 0:DH])
            else:
                T.copy(q, qs)
                T.gemm(qs, ks[:, 0:DH], sf, transpose_B=True)
            T.copy(sf, o)
    return main


if __name__ == "__main__":
    torch.manual_seed(0)
    q = torch.randn(GQA, DH, device="cuda", dtype=torch.float32)
    k = torch.randn(ABLK, KVROW, device="cuda", dtype=torch.bfloat16)
    refs = {"shared": q.to(torch.bfloat16).float() @ k[:, :DH].float().t(),
            "fragment": q.to(torch.bfloat16).float() @ k[:, :DH].float().t(),
            "pv": q.to(torch.bfloat16).float() @ k[:DH, :DH].float()}
    for where in ("shared", "fragment", "pv"):
        ref = refs[where]
        o = torch.zeros(GQA, ABLK, device="cuda", dtype=torch.float32)
        try:
            build(where)(q, k, o)
            torch.cuda.synchronize()
            got = o[:, :DH] if where == "pv" else o
            rel = ((got - ref).norm() / ref.norm()).item()
            print(f"A in {where:9s} {'OK   ' if rel < 3e-3 else 'WRONG'} rel_l2 {rel:.3e}")
        except Exception as error:
            print(f"A in {where:9s} FAIL\n" + "\n".join(l for l in str(error).splitlines() if l.strip() and not l.startswith(" "))[:700])
