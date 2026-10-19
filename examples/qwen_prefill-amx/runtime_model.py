"""Python runtime twins with native AMX projections and Torch host operations."""

import torch
import torch.nn.functional as F
from tilefoundry.runtime import runtime_func, runtime_module
from tilefoundry.dsl import ModuleList
from kernels import ops
from scheduled import make_scheduled


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


def make_runtime(config, sequence=128):
    semantic = make_scheduled(config, sequence)
    eps = config["rms_norm_eps"]
    hq = config["num_attention_heads"]
    hk = config["num_key_value_heads"]
    d = config["head_dim"]
    gated = config.get("attn_output_gate", False)

    def linear(sem, *, head=False):
        @runtime_module(sem)
        class Linear:
            @runtime_func
            def forward(self, x, w):
                if head:
                    return x.half().float() @ w.float()
                out = torch.empty(x.shape[0], w.shape[0] * w.shape[2])
                ops().linear(x.contiguous(), w, out)
                return out

        return Linear

    def make_norm(sem):
        @runtime_module(sem)
        class Norm:
            @runtime_func
            def forward(self, x, gamma):
                return F.rms_norm(x, (x.shape[-1],), gamma, eps)

        return Norm

    def make_mlp(sem):
        @runtime_module(sem)
        class MLP:
            gate = linear(sem.gate)
            up = linear(sem.up)
            down = linear(sem.down)

            @runtime_func
            def combine(self, g, u):
                return F.silu(g) * u

            @runtime_func
            def forward(self, x):
                return self.down(self.combine(self.gate(x), self.up(x)))

        return MLP

    def attention(sem):
        @runtime_module(sem.query_norm)
        class QueryNorm:
            @runtime_func
            def query_norm(self, q, gamma_q):
                return F.rms_norm(q, (d,), gamma_q, eps)

        @runtime_module(sem.key_norm)
        class KeyNorm:
            @runtime_func
            def key_norm(self, k, gamma_k):
                return F.rms_norm(k, (d,), gamma_k, eps)

        def rope(x, cos, sin):
            r = cos.shape[-1]
            part = x[..., :r]
            turned = torch.cat((-part[..., r // 2 :], part[..., : r // 2]), -1)
            result = part * cos[None, :, None, :] + turned * sin[None, :, None, :]
            return torch.cat((result, x[..., r:]), -1)

        @runtime_module(sem)
        class Attention:
            query = linear(sem.query)
            key = linear(sem.key)
            value = linear(sem.value)
            output = linear(sem.output)
            query_norm = QueryNorm
            key_norm = KeyNorm

            @runtime_func
            def query_rope(self, q, cos, sin):
                return rope(q, cos, sin)

            @runtime_func
            def key_rope(self, k, cos, sin):
                return rope(k, cos, sin)

            @runtime_func
            def split_query(self, x):
                return x.reshape(1, sequence, hq, -1)[..., :d]

            @runtime_func
            def split_kv(self, x):
                return x.reshape(1, sequence, hk, d)

            if gated:

                @runtime_func
                def split_gate(self, x):
                    return x.reshape(sequence, hq, 2 * d)[..., d:].reshape(
                        sequence, hq * d
                    )

                @runtime_func
                def gate_output(self, x, gate):
                    return x * torch.sigmoid(gate)

            @runtime_func
            def attend(self, q, k, v, mask):
                q = q.transpose(1, 2)
                k = k.repeat_interleave(hq // hk, 2).permute(0, 2, 3, 1)
                v = v.repeat_interleave(hq // hk, 2).transpose(1, 2)
                p = torch.softmax(q @ k / (d**0.5) + mask, -1)
                return (p @ v).transpose(1, 2).reshape(sequence, hq * d)

            @runtime_func
            def forward(self, x, cos, sin, mask):
                qg = self.query(x)
                q = self.query_rope(self.query_norm(self.split_query(qg)), cos, sin)
                k = self.key_rope(self.key_norm(self.split_kv(self.key(x))), cos, sin)
                out = self.attend(q, k, self.split_kv(self.value(x)), mask)
                if gated:
                    out = self.gate_output(out, self.split_gate(qg))
                return self.output(out)

        return Attention

    def recurrent(sem):
        @runtime_module(sem.math)
        class GDN:
            @runtime_func
            def forward(self, qkv, z, alpha, beta_raw, conv, decay, dt_bias, gamma):
                return gdn(
                    qkv,
                    z,
                    alpha,
                    beta_raw,
                    conv,
                    decay,
                    dt_bias,
                    gamma,
                    nk=config["linear_num_key_heads"],
                    nv=config["linear_num_value_heads"],
                    dk=config["linear_key_head_dim"],
                    dv=config["linear_value_head_dim"],
                    eps=eps,
                )

        @runtime_module(sem)
        class Recurrent:
            qkv = linear(sem.qkv)
            gate = linear(sem.gate)
            alpha = linear(sem.alpha)
            beta = linear(sem.beta)
            output = linear(sem.output)
            math = GDN

            @runtime_func
            def forward(self, x, cos, sin, mask):
                return self.output(
                    self.math(self.qkv(x), self.gate(x), self.alpha(x), self.beta(x))
                )

        return Recurrent

    def layer(sem, kind):
        @runtime_module(sem)
        class Layer:
            norm_in = make_norm(sem.norm_in)
            norm_ffn = make_norm(sem.norm_ffn)
            mlp = make_mlp(sem.mlp)
            mixer = (
                attention(sem.mixer)
                if kind == "full_attention"
                else recurrent(sem.mixer)
            )

            @runtime_func
            def add_mixer(self, x, y):
                return x + y

            @runtime_func
            def add_mlp(self, x, y):
                return x + y

            @runtime_func
            def forward(self, x, cos, sin, mask):
                x = self.add_mixer(x, self.mixer(self.norm_in(x), cos, sin, mask))
                return self.add_mlp(x, self.mlp(self.norm_ffn(x)))

        return Layer

    kinds = config.get("layer_types", ["full_attention"] * config["num_hidden_layers"])
    definitions = {
        kind: layer(semantic.layers[kinds.index(kind)], kind) for kind in set(kinds)
    }

    @runtime_module(semantic)
    class QwenPrefill:
        layers = ModuleList(tuple(definitions[kind] for kind in kinds))
        norm = make_norm(semantic.norm)
        head = linear(semantic.head, head=True)

        @runtime_func
        def embed(self, tokens, table):
            return table[tokens].float()

        @runtime_func
        def last(self, x):
            return x[-1:]

        @runtime_func
        def prefill(self, tokens, cos, sin, mask, table):
            hidden = self.embed(tokens)
            for child in self.layers:
                hidden = child(hidden, cos, sin, mask)
            return self.head(self.last(self.norm(hidden)))

    return QwenPrefill


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
