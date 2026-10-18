"""One-to-one runtime twins of the whole authored/scheduled model tree.

AMX linear leaves launch the existing hand-written native kernel through TVM
FFI. All other leaves execute their typed tensor program or a checked GDN
implementation. Hand-written runtime twins follow the HIR function call tree.
There is no llama.cpp whole-model callback in this path and no codegen.
"""

import os
from pathlib import Path
import torch, tvm_ffi
from tilefoundry.evaluator import evaluate
from tilefoundry.runtime import runtime_func, runtime_module
from tilefoundry.dsl import ModuleList
from tilefoundry.ir.core import Call
from tilefoundry.ir.visitor import collect_exprs
from tilefoundry.ir.hir.schedule import ScheduleOp
from instructions import Fma16

R = Path(__file__).resolve().parent


def gdn(qkv, z, alpha, beta_raw, conv, decay, dt_bias, gamma, *, nk, nv, dk, dv, eps):
    import torch.nn.functional as F

    s = qkv.shape[0]
    kd = nk * dk
    vd = nv * dv
    mixed = (
        F.silu(
            F.conv1d(
                qkv.T.unsqueeze(0),
                conv.unsqueeze(1),
                padding=conv.shape[1] - 1,
                groups=qkv.shape[1],
            )[:, :, :s]
        )
        .squeeze(0)
        .T
    )
    q = mixed[:, :kd].reshape(s, nk, dk).repeat(1, nv // nk, 1)
    k = mixed[:, kd : 2 * kd].reshape(s, nk, dk).repeat(1, nv // nk, 1)
    q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6)
    k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    v = mixed[:, 2 * kd :].reshape(s, nv, dv)
    g = decay * F.softplus(alpha + dt_bias)
    beta = torch.sigmoid(beta_raw)
    state = torch.zeros(nv, dk, dv)
    out = torch.empty(s, nv, dv)
    for t in range(s):
        state = state * torch.exp(g[t]).reshape(nv, 1, 1)
        key = k[t].reshape(nv, dk, 1)
        delta = (v[t] - (state * key).sum(1)) * beta[t].reshape(nv, 1)
        state = state + key * delta.reshape(nv, 1, dv)
        out[t] = (state * (q[t].reshape(nv, dk, 1) / (dk**0.5))).sum(1)
    out = out * torch.rsqrt((out * out).mean(-1, keepdim=True) + eps) * gamma
    return (out * F.silu(z.reshape(s, nv, dv))).reshape(s, vd)


def make_runtime(semantic, config, *, native=True):
    os.environ["TF_AMX_THREADS"] = "8"
    ffi = tvm_ffi.load_module(str(R / "artifacts/prefill.dylib")) if native else None
    stats = {"amx_calls": 0, "tensor_calls": 0}

    classes = {}

    def twin(mod):
        if id(mod) in classes:
            return classes[id(mod)]
        namespace = {
            key: ModuleList(tuple(twin(child) for child in slot.elements))
            if slot.indexed
            else twin(slot.elements[0])
            for key, slot in mod.children.items()
        }
        namespace["stats"] = stats
        for fn in mod.functions:
            is_amx = (
                len(fn.params) == 2
                and fn.params[1].name == "w"
                and len(fn.params[1].type.shape) == 3
                and any(
                    isinstance(v, Call)
                    and isinstance(v.target, ScheduleOp)
                    and isinstance(v.target.op, Fma16)
                    for v in collect_exprs(fn.body)
                )
            )

            def body(self, *args, fn=fn, is_amx=is_amx):
                if fn.name == "prefill":
                    tokens, cos, sin, mask = args[:4]
                    hidden = self.embed(tokens)
                    for child in self.layers:
                        hidden = child(hidden, cos, sin, mask)
                    return self.head(self.last(self.norm(hidden)))
                if fn.name == "forward" and self.children:
                    if hasattr(self, "norm_in"):
                        x, cos, sin, mask = args
                        x = self.add_mixer(
                            x, self.mixer(self.norm_in(x), cos, sin, mask)
                        )
                        return self.add_mlp(x, self.mlp(self.norm_ffn(x)))
                    if hasattr(self, "down"):
                        x = args[0]
                        return self.down(self.combine(self.gate(x), self.up(x)))
                    if hasattr(self, "qkv"):
                        x = args[0]
                        return self.output(
                            self.math(
                                self.qkv(x), self.gate(x), self.alpha(x), self.beta(x)
                            )
                        )
                    if hasattr(self, "query"):
                        x, cos, sin, mask = args
                        qg = self.query(x)
                        q = self.query_rope(
                            self.query_norm(self.split_query(qg)), cos, sin
                        )
                        k = self.key_rope(
                            self.key_norm(self.split_kv(self.key(x))), cos, sin
                        )
                        v = self.split_kv(self.value(x))
                        out = self.attend(q, k, v, mask)
                        if config.get("attn_output_gate", False):
                            out = self.gate_output(out, self.split_gate(qg))
                        return self.output(out)
                if is_amx:
                    x, w = args
                    M, K = x.shape
                    N = w.shape[0] * w.shape[2]
                    if native:
                        stats["amx_calls"] += 1
                        y = torch.empty(M, N, dtype=torch.float32)
                        ffi.linear(x.contiguous(), w, y)
                        return y
                    raw = w.permute(0, 2, 1).reshape(N, K)
                    return x.half().float() @ raw.float().T
                if (
                    any(p.name == "conv" for p in fn.params)
                    and len(args) == 8
                    and "linear_num_key_heads" in config
                ):
                    return gdn(
                        *args,
                        nk=config["linear_num_key_heads"],
                        nv=config["linear_num_value_heads"],
                        dk=config["linear_key_head_dim"],
                        dv=config["linear_value_head_dim"],
                        eps=config["rms_norm_eps"],
                    )
                stats["tensor_calls"] += 1
                return evaluate(fn, *args)

            namespace[fn.name] = runtime_func(body)
        classes[id(mod)] = runtime_module(mod)(
            type(mod.name + "Runtime", (), namespace)
        )
        return classes[id(mod)]

    return twin(semantic)


def inputs(config, sequence, tokens):
    r = int(
        config["head_dim"]
        * config.get("rope_parameters", {}).get("partial_rotary_factor", 1)
    )
    theta = config.get("rope_parameters", {}).get(
        "rope_theta", config.get("rope_theta", 1000000)
    )
    inv = 1 / (theta ** (torch.arange(0, r, 2, dtype=torch.float32) / r))
    freq = torch.arange(sequence, dtype=torch.float32)[:, None] * inv[None, :]
    freq = torch.cat([freq, freq], dim=-1)
    mask = torch.full((sequence, sequence), float("-inf")).triu(1)
    return tokens.long(), freq.cos(), freq.sin(), mask
