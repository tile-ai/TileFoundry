"""Tiny migration smoke: prepare/load/native run against the model HIR."""

from pathlib import Path
import torch
from tilefoundry.runtime import DictResource, SafetensorsResource
from model import build
from scheduled import make_scheduled
from runtime_model import make_runtime, inputs

ROOT = Path(__file__).resolve().parent
CONFIG = dict(
    hidden_size=64,
    intermediate_size=128,
    vocab_size=64,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    rms_norm_eps=1e-6,
    num_hidden_layers=2,
    rope_theta=10000,
)


def raw_weights(mod, prefix="", out=None):
    out = {} if out is None else out
    for fn in mod.functions:
        converters = dict(fn.converters)
        for param in fn.params:
            if not param.is_const:
                continue
            for p in (
                converters[param.name].params if param.name in converters else (param,)
            ):
                key = prefix + p.name
                if key in out:
                    continue
                x = torch.randn(p.type.shape) * 0.025
                if p.name.startswith("gamma"):
                    x = 1 + x
                if p.name == "decay":
                    x = -x.abs() - 0.01
                out[key] = x.half() if p.type.dtype.name == "f16" else x.float()
    for key, slot in mod.children.items():
        for i, ch in enumerate(slot.elements):
            raw_weights(
                ch, prefix + key + "." + (str(i) + "." if slot.indexed else ""), out
            )
    return out


if __name__ == "__main__":
    torch.set_num_threads(8)
    torch.manual_seed(17)
    for name in ("dense", "hybrid"):
        c = dict(CONFIG)
        if name == "hybrid":
            c.update(
                attn_output_gate=True,
                layer_types=["linear_attention", "full_attention"],
                linear_num_key_heads=16,
                linear_num_value_heads=32,
                linear_key_head_dim=4,
                linear_value_head_dim=4,
                linear_conv_kernel_dim=4,
                rope_parameters={"partial_rotary_factor": 0.5, "rope_theta": 10000},
            )
        plain = build(c, 32)
        scheduled = make_scheduled(c, 32)
        raw = DictResource(raw_weights(plain))
        outputs = []
        args = inputs(c, 32, torch.arange(32) % 64)
        for label, sem in [("plain", plain), ("scheduled", scheduled)]:
            p = ROOT / "artifacts" / ("migration-check-" + name + "-" + label)
            sem.prepare(raw, str(p), device="cpu")
            resource = SafetensorsResource(str(p), device="cpu")
            if label == "plain":
                result = sem.load(resource)(*args)
            else:
                runtime = make_runtime(sem, c)()
                runtime.load(resource)
                result = runtime(*args)
                assert runtime.stats["amx_calls"] > 0
            assert torch.isfinite(result).all()
            outputs.append(result)
        torch.testing.assert_close(outputs[1], outputs[0], rtol=2e-4, atol=2e-4)
        print(
            "PASS",
            name,
            "native runtime vs whole-model HIR",
            float((outputs[1] - outputs[0]).abs().max()),
            flush=True,
        )
