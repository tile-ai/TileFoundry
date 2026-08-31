"""Mega against the op-by-op path across the block boundary the context splits on.

The attention stripe is `ABLK`-sized, so the interesting lengths are the ones
where a step first fills a block, first spills into a second, and where several
CTAs have stripes of their own.
"""
import os, pathlib, sys, torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import paths
os.environ["NEMO_IMPL"] = "mega"
from tilefoundry.runtime import SafetensorsResource
import runtime_model as rm, model
DEV = "cuda:0"
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 300
CHECK = {0, 1, 2, 126, 127, 128, 129, 255, 256, 257, STEPS - 1}
CHECK |= {int(x) for x in sys.argv[2].split(",")} if len(sys.argv) > 2 else set()
twin = rm.Nemotron35Lightning30BA3BRuntime()
twin.load(SafetensorsResource(str(paths.need("prepared")), device=DEV))
g = torch.Generator(device="cpu").manual_seed(0)
ids = torch.randint(0, model.V, (STEPS + 1,), generator=g).to(DEV)


def run(impl):
    rm.set_impl(impl)
    caches = twin.init_caches(device=DEV, capacity=STEPS + 8)
    out = {}
    for step in range(STEPS):
        args = twin.prepare_inputs_for_generation(ids[: step + 1], step, caches, device=DEV)
        logits, fresh = twin.forward(*args)
        caches = twin.append_cache(caches, fresh)
        if step in CHECK:
            out[step] = logits.reshape(-1).float().clone()
    return out


a, b = run("mega"), run("ops")
bad = 0
for k in sorted(a):
    rel = ((a[k] - b[k]).norm() / b[k].norm()).item()
    same = int(a[k].argmax()) == int(b[k].argmax())
    top = torch.topk(b[k], 2).values
    gap = float(top[0] - top[1])
    bad += not same
    print(f"ctx {k + 1:5d}  rel_l2 {rel:.3e}  argmax mega {int(a[k].argmax()):6d}"
          f" ops {int(b[k].argmax()):6d}  top-2 gap {gap:7.4f}"
          f"  {'OK' if same else 'DIFF'}")
print("all argmax agree" if not bad else f"{bad} disagree")
