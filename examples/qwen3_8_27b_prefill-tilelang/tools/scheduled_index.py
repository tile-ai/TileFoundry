"""The four-artifact chain, as data: for every model.py function, the
scheduled HIR module and the kernel functions `schedule finalize` lowers
(one TIR file each), and the runtime kernels that implement them.

    python tools/scheduled_index.py --jobs [PREFIX ...]   # "<source> <tir path>" lines; PREFIX matches e.g. QKV or QKV8192
    python tools/scheduled_index.py --table               # markdown table for BENCH.md
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

LENS = (1024, 4096, 8192)

# model.py function -> (scheduled file, module prefix, tir stem, kernel funcs, runtime kernels)
CHAIN = [
    ("Qwen3_8_27B.embed", "scheduled/embed_final.py", "Embed", "embed", ["embed"], "elementwise.embed_kernel"),
    ("Layer.input_norm / post_norm", "scheduled/norm.py", "RMSNorm", "rmsnorm", ["rmsnorm"], "elementwise.rmsnorm_kernel"),
    ("GatedDeltaNet.in_proj", "scheduled/gemm.py", "InProj", "in_proj", ["pack*", "whole"], "pack.in_proj, gemm.gemm_kernel(plain)"),
    ("GatedDeltaNet.prep", "scheduled/gdn_elementwise.py", "Prep", "prep",
     ["conv_q", "conv_k", "conv_v", "gate_g", "gate_beta"], "elementwise.prep_kernel"),
    ("GatedDeltaNet.delta_rule (1 of 2)", "scheduled/gdn.py", "DeltaRuleIntra", "delta_rule", ["intra_g", "intra_wup"],
     "gdn.intra_kernel"),
    ("GatedDeltaNet.delta_rule (2 of 2)", "scheduled/gdn.py", "DeltaRuleRecur", "delta_rule", ["recur_out"],
     "gdn.recur_out_kernel"),
    ("GatedDeltaNet.gated_norm", "scheduled/gdn_elementwise.py", "GatedNorm", "gated_norm", ["gated_norm"],
     "elementwise.gated_norm_kernel"),
    ("GatedDeltaNet.out_proj (1 of 2 where split)", "scheduled/gemm.py", "OutProj", "out_proj", ["pieces?", "whole"], "gemm.gemm_kernel(residual)"),
    ("GatedDeltaNet.out_proj (2 of 2)", "scheduled/gemm.py", "OutProjTail", "out_proj", ["tail?"], "gemm.gemm_tail_kernel(residual)"),
    ("Attention.qkv (1 of 2 where split)", "scheduled/gemm.py", "QKV", "qkv", ["pack*", "pieces?", "whole", "whole_b?", "whole_last?"],
     "pack.qkv, gemm.gemm_kernel(qk_norm_rope)"),
    ("Attention.qkv (2 of 2)", "scheduled/gemm.py", "QKVTail", "qkv", ["tail?"], "gemm.gemm_tail_kernel(qk_norm_rope)"),
    ("Attention.attention", "scheduled/attention.py", "Attention", "attention", ["attention"],
     "attention.attention_tir_kernel (attention_ws_kernel at n = 128)"),
    ("Attention.o_proj (1 of 2 where split)", "scheduled/gemm.py", "OProj", "o_proj", ["pieces?", "whole"], "gemm.gemm_kernel(residual)"),
    ("Attention.o_proj (2 of 2)", "scheduled/gemm.py", "OProjTail", "o_proj", ["tail?"], "gemm.gemm_tail_kernel(residual)"),
    ("MLP.gate_up (1 of 2 where split)", "scheduled/gemm.py", "GateUp", "gate_up", ["pack*", "pieces?", "whole"], "pack.gate_up, gemm.gemm_kernel(swiglu)"),
    ("MLP.gate_up (2 of 2)", "scheduled/gemm.py", "GateUpTail", "gate_up", ["tail?"], "gemm.gemm_tail_kernel(swiglu)"),
    ("MLP.down (1 of 2 where split)", "scheduled/gemm.py", "Down", "down", ["pieces?", "whole"], "gemm.gemm_kernel(residual)"),
    ("MLP.down (2 of 2)", "scheduled/gemm.py", "DownTail", "down", ["tail?"], "gemm.gemm_tail_kernel(residual)"),
    ("Qwen3_8_27B.final", "scheduled/embed_final.py", "Final", "final", ["final"], "attention.final_kernel"),
]


def _has_tail(prefix, n):
    from scheduled import gemm as S
    prefix = prefix.removesuffix("Tail")
    K = {"Down": S.I, "OutProj": 6144, "OProj": 6144}.get(prefix, S.H)
    N = {"GateUp": 2 * S.I, "InProj": 16640, "QKV": S.QKV_W}.get(prefix, S.H)
    return S._plan(n, N, K)["REM"] > 0


def _defined(src, mod, n, name):
    """Does the length-n module define `name`? (`whole_b` / `whole_last` exist at some lengths only.)"""
    import importlib
    m = importlib.import_module(src.removesuffix(".py").replace("/", "."))
    return hasattr(getattr(m, f"{mod}{n}", None), name)


def jobs(prefixes):
    for _fn, src, mod, stem, kernels, _rt in CHAIN:
        if not Path(src).exists():
            continue
        for n in LENS:
            if prefixes and not any(f"{mod}{n}".startswith(p) for p in prefixes):
                continue
            for k in kernels:
                name = k.rstrip("*?")
                if k.endswith("*"):            # weight-only: once, from the 1024 module
                    if n == 1024:
                        yield f"{src}:{mod}{n}.{name}", f"tir/{stem}_{name}.py"
                    continue
                if k.endswith("?") and not (_has_tail(mod, n) and _defined(src, mod, n, name)):
                    continue
                tail = "" if name == stem else f"_{name}"
                yield f"{src}:{mod}{n}.{name}", f"tir/{stem}{tail}_{n}.py"


def table():
    print("| model.py function | scheduled HIR (module per length) | kernel @funcs -> tir/ | runtime kernels |")
    print("|---|---|---|---|")
    def label(src, mod, k):
        name = k.rstrip("*?")
        if k.endswith("*"):
            return f"{name} (once)"
        if k.endswith("?"):
            ns = [n for n in LENS if _has_tail(mod, n) and _defined(src, mod, n, name)]
            return f"{name} ({', '.join(map(str, ns))})"
        return name

    for fn, src, mod, stem, kernels, rt in CHAIN:
        ks = ", ".join(label(src, mod, k) for k in kernels)
        print(f"| `{fn}` | `{src}`: `{mod}{{n}}` | {ks} -> `tir/{stem}_*.py` | `{rt}` |")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--table"]:
        table()
    else:
        for src, out in jobs(sys.argv[2:]):
            print(src, out)
