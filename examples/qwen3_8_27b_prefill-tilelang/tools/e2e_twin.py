"""The runtime twin end to end against transformers: first token and logits error.

    CUDA_VISIBLE_DEVICES=1 python tools/e2e_twin.py --lens 1024,4096,8192 [--graph]
"""
import argparse, sys, time
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
import paths  # noqa: E402
REFDATA = paths.need("refdata")
import run as RUN  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--lens", default="1024")
ap.add_argument("--graph", action="store_true")
args = ap.parse_args()
t0 = time.perf_counter()
twin = RUN.load_twin(paths.need("ckpt"))
print(f"loaded twin in {time.perf_counter() - t0:.0f}s", flush=True)
for n in [int(x) for x in args.lens.split(",")]:
    ref = torch.load(REFDATA / f"n{n}" / "logits.pt")
    ids = ref["ids"].cuda()
    t = time.perf_counter()
    if args.graph:
        pre = RUN.Prefill(twin, n)
        logits, token = pre(ids, n - 1)
    else:
        with torch.no_grad():
            logits = twin.forward(ids, torch.tensor([n - 1], device="cuda"))
    torch.cuda.synchronize()
    hl = ref["logits"].cuda().float()
    lg = logits[0].float()
    first = int(torch.argmax(lg))
    d = (lg - hl).abs()
    print(f"n={n} twin_first={first} hf_first={ref['first_token']} {'SAME' if first == ref['first_token'] else 'DIFFERENT'} "
          f"max_abs_logit_err={d.max().item():.4f} rel_l2={(d.norm() / hl.norm()).item():.3e} "
          f"(first run incl. compile {time.perf_counter() - t:.0f}s)", flush=True)
