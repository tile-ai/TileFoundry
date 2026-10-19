"""Canonical GGUF resource aliases for the whole Prefill HIR tree."""

from pathlib import Path
import numpy as np
import torch
from gguf import GGUFReader, GGMLQuantizationType


def aliases(config):
    out = {
        "table": "token_embd.weight",
        "norm.gamma": "output_norm.weight",
        "head.raw": "token_embd.weight",
    }
    for i, kind in enumerate(
        config.get("layer_types", ["full_attention"] * config["num_hidden_layers"])
    ):
        p = f"layers.{i}"
        b = f"blk.{i}"
        out[p + ".norm_in.gamma"] = b + ".attn_norm.weight"
        out[p + ".norm_ffn.gamma"] = b + (
            ".post_attention_norm.weight"
            if "linear_num_key_heads" in config
            else ".ffn_norm.weight"
        )
        for name in ("gate", "up", "down"):
            out[p + ".mlp." + name + ".raw"] = b + ".ffn_" + name + ".weight"
        if kind == "full_attention":
            for name, tensor in (
                ("query", "attn_q"),
                ("key", "attn_k"),
                ("value", "attn_v"),
                ("output", "attn_output"),
            ):
                out[p + ".mixer." + name + ".raw"] = b + "." + tensor + ".weight"
            out[p + ".mixer.query_norm.gamma_q"] = b + ".attn_q_norm.weight"
            out[p + ".mixer.key_norm.gamma_k"] = b + ".attn_k_norm.weight"
        else:
            for name, tensor in (
                ("qkv", "attn_qkv"),
                ("gate", "attn_gate"),
                ("alpha", "ssm_alpha"),
                ("beta", "ssm_beta"),
                ("output", "ssm_out"),
            ):
                out[p + ".mixer." + name + ".raw"] = b + "." + tensor + ".weight"
            for name, tensor in (
                ("conv", "ssm_conv1d.weight"),
                ("decay", "ssm_a"),
                ("dt_bias", "ssm_dt.bias"),
                ("gamma", "ssm_norm.weight"),
            ):
                out[p + ".mixer.math." + name] = b + "." + tensor
    return out


class GGUFResource:
    def __init__(self, path, alias, *, prefix="", shared=None):
        self.path = Path(path)
        self.alias = alias
        self.prefix = prefix
        self.shared = (
            shared
            if shared is not None
            else {"reader": GGUFReader(self.path), "cache": {}}
        )
        self.tensors = {t.name: t for t in self.shared["reader"].tensors}

    def load(self, name):
        raw = self.alias[self.prefix + name]
        cache = self.shared["cache"]
        if raw in cache:
            return cache[raw]
        t = self.tensors[raw]
        if t.tensor_type == GGMLQuantizationType.Q8_0:
            k, n = map(int, t.shape)
            blocks = np.asarray(t.data).reshape(n, k // 32, 34)
            codes = blocks[:, :, 2:].copy().view(np.int8).astype(np.float32)
            scale = (
                blocks[:, :, :2]
                .copy()
                .view(np.float16)
                .reshape(n, k // 32, 1)
                .astype(np.float32)
            )
            value = torch.from_numpy((codes * scale).reshape(n, k).astype(np.float16))
        elif t.tensor_type == GGMLQuantizationType.F32:
            value = torch.from_numpy(
                np.asarray(t.data).copy().reshape(tuple(map(int, t.shape[::-1])))
            ).float()
        else:
            raise ValueError(f"Unsupported source dtype {t.name}: {t.tensor_type}")
        cache.clear()
        cache[raw] = value
        return value

    def load_group(self, name):
        return None

    def subtree(self, seg):
        return GGUFResource(
            self.path, self.alias, prefix=self.prefix + seg + ".", shared=self.shared
        )


class PreparedModelResource:
    """GGUF checkpoint with the original N/64,K,64 prepared AMX weights."""

    def __init__(self, checkpoint, directory):
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        self.directory = Path(directory).expanduser().resolve()
        self.reader = GGUFReader(str(self.checkpoint))

    def entries(self):
        for tensor in self.reader.tensors:
            if (
                tensor.tensor_type != GGMLQuantizationType.Q8_0
                or len(tensor.shape) != 2
            ):
                continue
            k, n = map(int, tensor.shape)
            if n % 64 or k % 32:
                continue
            if not tensor.name or any(
                c
                not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_."
                for c in tensor.name
            ):
                raise ValueError("Invalid GGUF tensor name")
            yield tensor.name, n, k, self.directory / (tensor.name + ".bin")

    def prepare(self):
        from scheduled import scheduled_projection
        from tilefoundry.runtime import DictResource, SafetensorsResource

        self.directory.mkdir(parents=True, exist_ok=True)
        raw = GGUFResource(
            self.checkpoint, {t.name: t.name for t in self.reader.tensors}
        )
        for name, n, k, path in self.entries():
            if path.exists():
                if path.stat().st_size != n * k * 2:
                    raise ValueError(f"Invalid prepared weight size: {path}")
                continue
            stage = self.directory / name
            scheduled_projection(128, n, k).prepare(
                DictResource({"raw": raw.load(name)}), str(stage), device="cpu"
            )
            packed = SafetensorsResource(str(stage), device="cpu").load("w")
            temporary = path.with_suffix(".bin.tmp")
            packed.contiguous().numpy().tofile(temporary)
            temporary.replace(path)
