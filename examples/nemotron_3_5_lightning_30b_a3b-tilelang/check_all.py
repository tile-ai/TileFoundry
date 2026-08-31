"""Run `tilefoundry check` over every output of the step, with derived bounds.

There is no default tolerance and there should not be one, so each of the 59
outputs gets a bound argued from how many times a value on the way to it lands
in bf16. One round-to-nearest is at most 2^-9 relative; the landings on the path
to a layer-i output are counted below; and the bound is the random-walk sum,
2^-9 * sqrt(k), which is the aggregate `rel_l2` measures.

The two outputs that are not arithmetic get the bound that fits them instead:
the convolution window a step hands on is a shifted slice of the window it was
handed, so it is compared bitwise.
"""
from __future__ import annotations

import argparse
import math
import pathlib
import subprocess
import sys

import model
import paths

#: bf16 landings a layer of each kind puts on the path through it, counted off
#: this implementation rather than guessed. Pre-norm 2 (the normalised value
#: lands before the weight multiplies it, which is what the checkpoint's own
#: RMSNorm does) and the residual add 1 are common; the rest is the mixer's:
#:
#:   mamba  in_proj 1 + conv 2 (bias, then silu) + scan 3 (the two inside dbx,
#:          then the C contraction) + y with the D skip 3 + gated norm 2
#:          + out_proj 1                                              = 12
#:   attn   qkv 1 + the context divide 1 + o_proj 1                   =  3
#:   moe    up 2 (relu, then the square) + down 1 + shared 3
#:          + the routed/shared combine 3                             =  9
_LANDINGS = {"linear_attention": 2 + 1 + 12, "full_attention": 2 + 1 + 3,
             "moe": 2 + 1 + 9}
_ULP = 2.0 ** -9


def outputs():
    """(name, kind, landings on the path to it), in the order the step returns."""
    out, k = [("logits", "logits", 0)], 0
    for i, kind in enumerate(model.LAYER_KINDS):
        if kind == "linear_attention":
            # the pre-norm's two, the input projection's one, the convolution's
            # two, and the scan's own
            # The window a step hands on is two columns of the window it was
            # handed, shifted, plus one fresh column from the input projection:
            # pre-norm 2 and in_proj 1. Two thirds of it is a copy and one third
            # is arithmetic, so the bound is the projection's, not `equal`.
            # The SSM state adds the convolution's 2 and the recurrence's 2.
            out += [(f"l{i}_conv_out", "state", k + 3), (f"l{i}_ssm_out", "state", k + 7)]
        elif kind == "full_attention":
            out += [(f"l{i}_k_new", "kv", k + 3), (f"l{i}_v_new", "kv", k + 3)]
        k += _LANDINGS[kind]
    out[0] = ("logits", "logits", k + 2 + 1)      # the closing norm and the head
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default=None,
                    help="what `tilefoundry check --ckpt` wants is the "
                         "**prepared** directory (or $NEMOTRON35_PREPARED)")
    ap.add_argument("--source", default="runtime_model.py:Nemotron35Lightning30BA3BRuntime.decode_step")
    ap.add_argument("--ctx-full", default="0")
    ap.add_argument("--ctx-tail", default="1")
    ap.add_argument("--json", default="reports/check.json")
    ap.add_argument("--acts", default=None,
                    help="a directory of real activation files from dump_acts.py")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()

    cmd = ["tilefoundry", "check", a.source, "--ckpt", str(paths.need("prepared", a.ckpt)),
           "--dim", f"ctx_full={a.ctx_full}", "--dim", f"ctx_tail={a.ctx_tail}",
           "--json", a.json]
    cmd += [] if a.acts else ["--inputs", "real"]
    if a.acts:
        for f in sorted(pathlib.Path(a.acts).glob("*.pt")):
            cmd += ["--input", str(f)]
    for at, (name, kind, k) in enumerate(outputs()):
        bound = _ULP * math.sqrt(max(k, 1))
        cmd += ["--out", f"output[{at}]", "--fn", "nan_inf"]
        cmd += ["--fn", "rel_l2", "--max", f"{bound:.4g}"]
        if kind == "logits":
            cmd += ["--fn", "cosine", "--min", "0.999"]
    print(f"# {len(outputs())} outputs, "
          f"bounds 2^-9*sqrt(k) with k from {1} to {max(k for _, _, k in outputs())}")
    if a.dry:
        print(" ".join(cmd))
        return 0
    return subprocess.call([sys.executable.replace("python", "tilefoundry")] + cmd[1:]
                           if False else cmd)


if __name__ == "__main__":
    raise SystemExit(main())
