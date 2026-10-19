"""Whole text-model Prefill HIR, with all layers and checkpoint-owned weights.

Every compute and composition step is an authored @func, including the whole
Prefill entry. The entry lists static instance calls; layer bodies remain shared. GGUF's tiled GDN head order,
materialized RMS scales and negative exp(A_log) are retained explicitly.
"""

import math
from tilefoundry import module, func
from tilefoundry.dsl import Tensor, ConstTensor, ModuleList, Topology, tf
from tilefoundry.target import AmxTarget


def projection(M, N, K):
    @module(entry="forward")
    class Linear:
        @func
        def forward(x: Tensor[(M, K), "f32"], w: ConstTensor[(K, N), "f16"]):
            return tf.matmul(tf.cast(x, dtype="f16"), w, out_dtype="f32")

        @forward.converter("w")
        def _(raw: ConstTensor[(N, K), "f16"]):
            return tf.transpose(raw, perm=(1, 0))

    return Linear


def build(config, S, *, linear=projection):
    H = config["hidden_size"]
    F = config["intermediate_size"]
    V = config["vocab_size"]
    HQ = config["num_attention_heads"]
    HK = config["num_key_value_heads"]
    D = config["head_dim"]
    EPS = config["rms_norm_eps"]
    LAYERS = config["num_hidden_layers"]
    GATED = config.get("attn_output_gate", False)
    R = int(D * config.get("rope_parameters", {}).get("partial_rotary_factor", 1))
    Q = HQ * D
    KV = HK * D
    REPEAT = HQ // HK
    ATTN_SCALE = math.sqrt(D)

    @module(entry="forward")
    class Norm:
        @func
        def forward(x: Tensor[(S, H), "f32"], gamma: ConstTensor[(H,), "f32"]):
            return tf.rms_norm(x, gamma, eps=EPS)

    @module(entry="forward")
    class MLP:
        gate = ModuleList(linear(S, F, H))
        up = ModuleList(linear(S, F, H))
        down = ModuleList(linear(S, H, F))

        @func
        def combine(g: Tensor[(S, F), "f32"], u: Tensor[(S, F), "f32"]):
            return tf.silu(g) * u

        @func
        def forward(x: Tensor[(S, H), "f32"]):
            return down(combine(gate(x), up(x)))

    @module()
    class AttentionMath:
        @func
        def query_norm(
            q: Tensor[(1, S, HQ, D), "f32"], gamma_q: ConstTensor[(D,), "f32"]
        ):
            return tf.rms_norm(q, gamma_q, eps=EPS)

        @func
        def key_norm(
            k: Tensor[(1, S, HK, D), "f32"], gamma_k: ConstTensor[(D,), "f32"]
        ):
            return tf.rms_norm(k, gamma_k, eps=EPS)

        if R == D:

            @func
            def query_rope(
                q: Tensor[(1, S, HQ, D), "f32"],
                cos: Tensor[(S, R), "f32"],
                sin: Tensor[(S, R), "f32"],
            ):
                rot = q[:, :, :, :R]
                turned = tf.concat(
                    [-rot[:, :, :, R // 2 :], rot[:, :, :, : R // 2]], axis=-1
                )
                head = rot * tf.reshape(
                    cos, new_shape=(1, S, 1, R)
                ) + turned * tf.reshape(sin, new_shape=(1, S, 1, R))
                return head

            @func
            def key_rope(
                k: Tensor[(1, S, HK, D), "f32"],
                cos: Tensor[(S, R), "f32"],
                sin: Tensor[(S, R), "f32"],
            ):
                rot = k[:, :, :, :R]
                turned = tf.concat(
                    [-rot[:, :, :, R // 2 :], rot[:, :, :, : R // 2]], axis=-1
                )
                head = rot * tf.reshape(
                    cos, new_shape=(1, S, 1, R)
                ) + turned * tf.reshape(sin, new_shape=(1, S, 1, R))
                return head

        else:

            @func
            def query_rope(
                q: Tensor[(1, S, HQ, D), "f32"],
                cos: Tensor[(S, R), "f32"],
                sin: Tensor[(S, R), "f32"],
            ):
                rot = q[:, :, :, :R]
                turned = tf.concat(
                    [-rot[:, :, :, R // 2 :], rot[:, :, :, : R // 2]], axis=-1
                )
                head = rot * tf.reshape(
                    cos, new_shape=(1, S, 1, R)
                ) + turned * tf.reshape(sin, new_shape=(1, S, 1, R))
                return tf.concat([head, q[:, :, :, R:]], axis=-1)

            @func
            def key_rope(
                k: Tensor[(1, S, HK, D), "f32"],
                cos: Tensor[(S, R), "f32"],
                sin: Tensor[(S, R), "f32"],
            ):
                rot = k[:, :, :, :R]
                turned = tf.concat(
                    [-rot[:, :, :, R // 2 :], rot[:, :, :, : R // 2]], axis=-1
                )
                head = rot * tf.reshape(
                    cos, new_shape=(1, S, 1, R)
                ) + turned * tf.reshape(sin, new_shape=(1, S, 1, R))
                return tf.concat([head, k[:, :, :, R:]], axis=-1)

        @func
        def attend(
            q: Tensor[(1, S, HQ, D), "f32"],
            k: Tensor[(1, S, HK, D), "f32"],
            v: Tensor[(1, S, HK, D), "f32"],
            mask: Tensor[(S, S), "f32"],
        ):
            qe = tf.transpose(q, perm=(0, 2, 1, 3))
            ke = tf.transpose(
                tf.repeat_interleave(k, repeats=REPEAT, axis=2), perm=(0, 2, 3, 1)
            )
            ve = tf.transpose(
                tf.repeat_interleave(v, repeats=REPEAT, axis=2), perm=(0, 2, 1, 3)
            )
            score = tf.matmul(qe, ke, out_dtype="f32") / ATTN_SCALE + mask
            prob = tf.softmax(score, axis=-1)
            return tf.reshape(
                tf.transpose(tf.matmul(prob, ve, out_dtype="f32"), perm=(0, 2, 1, 3)),
                new_shape=(S, Q),
            )

    @module(entry="query_norm")
    class QueryNorm:
        query_norm = AttentionMath.query_norm

    @module(entry="key_norm")
    class KeyNorm:
        key_norm = AttentionMath.key_norm

    @module(entry="forward")
    class FullAttention:
        query = ModuleList(linear(S, Q * (2 if GATED else 1), H))
        key = ModuleList(linear(S, KV, H))
        value = ModuleList(linear(S, KV, H))
        output = ModuleList(linear(S, H, Q))
        query_norm = ModuleList(QueryNorm)
        key_norm = ModuleList(KeyNorm)
        query_rope = AttentionMath.query_rope
        key_rope = AttentionMath.key_rope
        attend = AttentionMath.attend
        if GATED:

            @func
            def split_query(x: Tensor[(S, 2 * Q), "f32"]):
                packed = tf.reshape(x, new_shape=(S, HQ, 2 * D))
                indices = tf.arange(Tensor[(D,), "i64"], start=0)
                return tf.reshape(
                    tf.index_select(packed, indices, dim=2), new_shape=(1, S, HQ, D)
                )

            @func
            def split_gate(x: Tensor[(S, 2 * Q), "f32"]):
                packed = tf.reshape(x, new_shape=(S, HQ, 2 * D))
                indices = tf.arange(Tensor[(D,), "i64"], start=D)
                return tf.reshape(
                    tf.index_select(packed, indices, dim=2), new_shape=(S, Q)
                )

            @func
            def gate_output(x: Tensor[(S, Q), "f32"], gate: Tensor[(S, Q), "f32"]):
                return x * tf.sigmoid(gate)
        else:

            @func
            def split_query(x: Tensor[(S, Q), "f32"]):
                return tf.reshape(x, new_shape=(1, S, HQ, D))

        @func
        def split_kv(x: Tensor[(S, KV), "f32"]):
            return tf.reshape(x, new_shape=(1, S, HK, D))

        if GATED:

            @func
            def forward(
                x: Tensor[(S, H), "f32"],
                cos: Tensor[(S, R), "f32"],
                sin: Tensor[(S, R), "f32"],
                mask: Tensor[(S, S), "f32"],
            ):
                qg = query(x)
                q = query_rope(query_norm(split_query(qg)), cos, sin)
                k = key_rope(key_norm(split_kv(key(x))), cos, sin)
                v = split_kv(value(x))
                out = attend(q, k, v, mask)
                return output(gate_output(out, split_gate(qg)))
        else:

            @func
            def forward(
                x: Tensor[(S, H), "f32"],
                cos: Tensor[(S, R), "f32"],
                sin: Tensor[(S, R), "f32"],
                mask: Tensor[(S, S), "f32"],
            ):
                qg = query(x)
                q = query_rope(query_norm(split_query(qg)), cos, sin)
                k = key_rope(key_norm(split_kv(key(x))), cos, sin)
                v = split_kv(value(x))
                out = attend(q, k, v, mask)
                return output(out)

    if "linear_num_key_heads" in config:
        NK = config["linear_num_key_heads"]
        NV = config["linear_num_value_heads"]
        DK = config["linear_key_head_dim"]
        DV = config["linear_value_head_dim"]
        CK = config["linear_conv_kernel_dim"]
        KD = NK * DK
        VD = NV * DV
        CV = 2 * KD + VD
        VR = NV // NK
        GDN_SCALE = math.sqrt(DK)

        @module(entry="forward")
        class GDNMath:
            @func
            def forward(
                qkv: Tensor[(S, CV), "f32"],
                z: Tensor[(S, VD), "f32"],
                alpha: Tensor[(S, NV), "f32"],
                beta_raw: Tensor[(S, NV), "f32"],
                conv: ConstTensor[(CV, CK), "f32"],
                decay: ConstTensor[(NV,), "f32"],
                dt_bias: ConstTensor[(NV,), "f32"],
                gamma: ConstTensor[(DV,), "f32"],
            ):
                padded = tf.concat([tf.zeros(Tensor[(CK - 1, CV), "f32"]), qkv], axis=0)
                mixed = tf.zeros(Tensor[(S, CV), "f32"])
                for t in range(S):
                    window = padded[t : t + CK, :]
                    row = tf.silu(
                        tf.reduce(
                            window * tf.transpose(conv, perm=(1, 0)),
                            axes=(0,),
                            keepdim=True,
                            kind="sum",
                        )
                    )
                    mixed = tf.insert_slice(mixed, row, (t, 0))
                q = tf.reshape(mixed[:, :KD], new_shape=(S, NK, DK))
                k = tf.reshape(mixed[:, KD : 2 * KD], new_shape=(S, NK, DK))
                # GGUF uses tiled V-head order. Repeat the K-head bank, rather
                # than duplicating each K head consecutively (HF's ordering).
                q = tf.reshape(
                    tf.repeat_interleave(
                        tf.reshape(q, new_shape=(S, 1, NK, DK)), repeats=VR, axis=1
                    ),
                    new_shape=(S, NV, DK),
                )
                k = tf.reshape(
                    tf.repeat_interleave(
                        tf.reshape(k, new_shape=(S, 1, NK, DK)), repeats=VR, axis=1
                    ),
                    new_shape=(S, NV, DK),
                )
                q = q * tf.rsqrt(
                    tf.reduce(tf.square(q), axes=(-1,), keepdim=True, kind="sum") + 1e-6
                )
                k = k * tf.rsqrt(
                    tf.reduce(tf.square(k), axes=(-1,), keepdim=True, kind="sum") + 1e-6
                )
                v = tf.reshape(mixed[:, 2 * KD :], new_shape=(S, NV, DV))
                g = decay * tf.softplus(alpha + dt_bias)
                beta = tf.sigmoid(beta_raw)
                state = tf.zeros(Tensor[(NV, DK, DV), "f32"])
                out = tf.zeros(Tensor[(S, NV, DV), "f32"])
                for t in range(S):
                    state = state * tf.reshape(
                        tf.exp(g[t : t + 1, :]), new_shape=(NV, 1, 1)
                    )
                    key = tf.reshape(k[t : t + 1, :, :], new_shape=(NV, DK, 1))
                    recall = tf.reduce(
                        state * key, axes=(1,), keepdim=False, kind="sum"
                    )
                    delta = (
                        tf.reshape(v[t : t + 1, :, :], new_shape=(NV, DV)) - recall
                    ) * tf.reshape(beta[t : t + 1, :], new_shape=(NV, 1))
                    state = state + key * tf.reshape(delta, new_shape=(NV, 1, DV))
                    query = (
                        tf.reshape(q[t : t + 1, :, :], new_shape=(NV, DK, 1))
                        / GDN_SCALE
                    )
                    read = tf.reshape(
                        tf.reduce(state * query, axes=(1,), keepdim=False, kind="sum"),
                        new_shape=(1, NV, DV),
                    )
                    out = tf.insert_slice(out, read, (t, 0, 0))
                gated = tf.rms_norm(out, gamma, eps=EPS) * tf.silu(
                    tf.reshape(z, new_shape=(S, NV, DV))
                )
                return tf.reshape(gated, new_shape=(S, VD))

        @module(entry="forward")
        class LinearAttention:
            qkv = ModuleList(linear(S, CV, H))
            gate = ModuleList(linear(S, VD, H))
            alpha = ModuleList(linear(S, NV, H))
            beta = ModuleList(linear(S, NV, H))
            output = ModuleList(linear(S, H, VD))
            math = ModuleList(GDNMath)

            @func
            def forward(
                x: Tensor[(S, H), "f32"],
                cos: Tensor[(S, R), "f32"],
                sin: Tensor[(S, R), "f32"],
                mask: Tensor[(S, S), "f32"],
            ):
                return output(math(qkv(x), gate(x), alpha(x), beta(x)))

    def layer(kind):
        mixer_type = FullAttention if kind == "full_attention" else LinearAttention

        @module(entry="forward")
        class Layer:
            norm_in = ModuleList(Norm)
            mixer = ModuleList(mixer_type)
            norm_ffn = ModuleList(Norm)
            mlp = ModuleList(MLP)

            @func
            def add_mixer(x: Tensor[(S, H), "f32"], y: Tensor[(S, H), "f32"]):
                return x + y

            @func
            def add_mlp(x: Tensor[(S, H), "f32"], y: Tensor[(S, H), "f32"]):
                return x + y

            @func
            def forward(
                x: Tensor[(S, H), "f32"],
                cos: Tensor[(S, R), "f32"],
                sin: Tensor[(S, R), "f32"],
                mask: Tensor[(S, S), "f32"],
            ):
                x = add_mixer(x, mixer(norm_in(x), cos, sin, mask))
                return add_mlp(x, mlp(norm_ffn(x)))

        return Layer

    kinds = config.get("layer_types", ["full_attention"] * LAYERS)

    definitions = {kind: layer(kind) for kind in set(kinds)}

    @module(target=AmxTarget(), topologies=(Topology("core", 8), Topology("amx", 1)))
    class Base:
        layers = ModuleList(tuple(definitions[kind] for kind in kinds))
        norm = ModuleList(Norm)
        head = ModuleList(linear(1, V, H))

        @func
        def embed(tokens: Tensor[(S,), "i64"], table: ConstTensor[(V, H), "f16"]):
            return tf.cast(tf.index_select(table, tokens, dim=0), dtype="f32")

        @func
        def last(x: Tensor[(S, H), "f32"]):
            return x[S - 1 : S, :]

    # Static slot indices support the heterogeneous GDN/attention stack.
    # Definitions stay shared: this emits calls, never cloned layer bodies.
    from pathlib import Path
    import hashlib

    lines = [
        "@module(entry='prefill',target=AmxTarget(),topologies=(Topology('core',8),Topology('amx',1)))",
        "class QwenPrefill:",
        "    layers=ModuleList(DEFINITIONS)",
        "    norm=ModuleList(NORM)",
        "    head=ModuleList(HEAD)",
        "    embed=BASE.embed",
        "    last=BASE.last",
        "    @func",
        "    def prefill(tokens:Tensor[(S,),'i64'],cos:Tensor[(S,R),'f32'],sin:Tensor[(S,R),'f32'],mask:Tensor[(S,S),'f32'],table:ConstTensor[(V,H),'f16']):",
        "        hidden=embed(tokens,table)",
    ]
    lines += [f"        hidden=layers[{i}](hidden,cos,sin,mask)" for i in range(LAYERS)]
    lines += ["        return head(last(norm(hidden)))"]
    source = "\n".join(lines) + "\n"
    dest = Path(__file__).parent / "artifacts/authored"
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / (
        hashlib.sha256((str(config) + source).encode()).hexdigest()[:16] + ".py"
    )
    path.write_text(source)
    namespace = dict(
        BASE=Base,
        DEFINITIONS=tuple(definitions[k] for k in kinds),
        NORM=Norm,
        HEAD=Base.head,
        ModuleList=ModuleList,
        module=module,
        func=func,
        Tensor=Tensor,
        ConstTensor=ConstTensor,
        AmxTarget=AmxTarget,
        Topology=Topology,
        S=S,
        R=R,
        H=H,
        V=V,
    )
    exec(compile(source, str(path), "exec"), namespace)
    return namespace["QwenPrefill"]


from config import config

Qwen3_1_7B = build(config("1.7b"), 128)
Qwen3_5_4B = build(config("4b"), 128)
