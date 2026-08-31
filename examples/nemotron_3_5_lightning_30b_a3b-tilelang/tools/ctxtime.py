"""Step time at a given context length, without decoding to get there.

The K/V cache is allocated at full capacity and zeroed by `init_caches`, and a
step reads it through two views cut by `caches["step"]`. Moving that counter is
therefore the same shape of work as having decoded that far -- the numbers in
the cache differ, the traffic and the loop trip counts do not -- which is what
lets a change be measured at 262144 in seconds instead of in an hour.
"""
import os, pathlib, sys, time, torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import paths
os.environ.setdefault("NEMO_IMPL", "mega")
from tilefoundry.runtime import SafetensorsResource
import runtime_model as rm, model
DEV = "cuda:0"
PTS = [int(x) for x in (sys.argv[1].split(",") if len(sys.argv) > 1
                        else "0,1024,16384,65536,262080".split(","))]
CAP = max(PTS) + 64
twin = rm.Nemotron35Lightning30BA3BRuntime()
twin.load(SafetensorsResource(str(paths.need("prepared")), device=DEV))
caches = twin.init_caches(device=DEV, capacity=CAP)
ids = torch.full((CAP,), 1784, dtype=torch.int64, device=DEV)
for ctx in PTS:
    caches["step"] = ctx
    for _ in range(12):
        args = twin.prepare_inputs_for_generation(ids[: ctx + 1], ctx, caches, device=DEV)
        twin.forward(*args)
    torch.cuda.synchronize()
    n = 50 if ctx <= 65536 else 20
    t0 = time.perf_counter()
    for _ in range(n):
        args = twin.prepare_inputs_for_generation(ids[: ctx + 1], ctx, caches, device=DEV)
        twin.forward(*args)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / n
    print(f"ctx {ctx:7d}  {dt * 1e3:8.3f} ms  {1 / dt:8.1f} tok/s", flush=True)
