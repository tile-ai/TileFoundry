"""Per-kernel GPU time of one prefill forward (CUDA graph replay), grouped by kernel name.

usage: python runtime/profile_forward.py T [--fake]   (--fake: machinery test mixers)"""
import collections, sys, torch
sys.path.insert(0, '.')
if '--fake' in sys.argv:
    sys.argv.remove('--fake')
    import runtime.experiments.test_machinery_fakes  # noqa: F401
from runtime.twin import Qwen3_8_27BTwin
from tilefoundry.runtime import SafetensorsResource
import prompting
from runtime import engine as eng

T = int(sys.argv[1])
tw = Qwen3_8_27BTwin(); tw.load(SafetensorsResource('ckpt', device='cuda'))
ids = torch.load(f'reference/T{T}/token_ids.pt').cuda(); cos, sin = prompting.rope_tables(T, eng.CFG, 'cuda')
for _ in range(3):
    tw(ids, cos, sin)
torch.cuda.synchronize()
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CUDA]) as prof:
    tw(ids, cos, sin); torch.cuda.synchronize()
agg = collections.defaultdict(lambda: [0.0, 0])
for ev in prof.events():
    if ev.device_type == torch.autograd.DeviceType.CUDA:
        name = ev.name.split('(')[0][:60]
        agg[name][0] += ev.device_time_total if hasattr(ev, 'device_time_total') else ev.cuda_time_total
        agg[name][1] += 1
tot = sum(v[0] for v in agg.values())
print(f"T={T} total kernel time {tot/1e3:.3f} ms over {sum(v[1] for v in agg.values())} kernels")
for name, (us, n) in sorted(agg.items(), key=lambda x: -x[1][0])[:25]:
    print(f"  {us/1e3:8.3f} ms {100*us/tot:5.1f}%  x{n:4d}  {name}")
