"""How many device launches one decode step costs, on each path."""
import os, pathlib, sys, torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import paths
os.environ["NEMO_IMPL"] = "mega"
from torch.profiler import profile, ProfilerActivity
from tilefoundry.runtime import SafetensorsResource
import runtime_model as rm
DEV = "cuda:0"
twin = rm.Nemotron35Lightning30BA3BRuntime()
twin.load(SafetensorsResource(str(paths.need("prepared")), device=DEV))
ids = torch.tensor([1784] * 32, device=DEV, dtype=torch.int64)


def count(impl, steps=3):
    rm.set_impl(impl)
    caches = twin.init_caches(device=DEV, capacity=64)
    for step in range(4):                    # warm, and get past the first block
        args = twin.prepare_inputs_for_generation(ids[: step + 1], step, caches, device=DEV)
        _, fresh = twin.forward(*args)
        caches = twin.append_cache(caches, fresh)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for k in range(steps):
            args = twin.prepare_inputs_for_generation(ids[: 5 + k], 4 + k, caches, device=DEV)
            _, fresh = twin.forward(*args)
            caches = twin.append_cache(caches, fresh)
        torch.cuda.synchronize()
    kern = memcpy = 0
    for e in prof.events():
        if str(getattr(e, "device_type", "")).endswith("CUDA"):
            name = e.name.lower()
            if "memcpy" in name or "memset" in name:
                memcpy += int(e.count)
            else:
                kern += int(e.count)
    print(f"{impl:5s}: {kern / steps:8.1f} kernel launches/step,"
          f" {memcpy / steps:5.1f} memcpy or memset/step")
    return kern / steps


for impl in ("mega", "ops"):
    count(impl)
