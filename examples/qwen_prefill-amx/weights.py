"""Canonical GGUF resource aliases for the whole Prefill HIR tree."""

from pathlib import Path
import numpy as np
import torch
from gguf import GGUFReader, GGMLQuantizationType


def explicit_weights(root):
    """Keep model weights on the root and pass them through shared definitions."""
    from dataclasses import replace
    from tilefoundry.ir.core import Call, Var
    from tilefoundry.ir.core.module import ModuleList
    from tilefoundry.ir.hir.function import Function
    from tilefoundry.ir.visitor import BindingSubstitutionCloner, collect_exprs
    from tilefoundry.visitor_registry.contexts import ValuationContext

    cache = {}

    def lift(mod):
        if id(mod) in cache:
            return cache[id(mod)]
        children = {}
        specs = {name: (ty, None, ()) for name, ty in mod.weights.items()}
        for fn in mod.functions:
            for name, converter in fn.converters:
                specs[name] = (specs[name][0], converter, ())
        for key, slot in mod.children.items():
            definitions = []
            for i, child in enumerate(slot.elements):
                definition, child_specs, _ = lift(child)
                definitions.append(definition)
                prefix = f"{key}__{i}__" if slot.indexed else f"{key}__"
                for name, (ty, converter, path) in child_specs.items():
                    specs[prefix + name] = (ty, converter, (prefix, *path))
            children[key] = ModuleList(
                tuple(definitions) if slot.indexed else definitions[0]
            )
        needs, functions = {}, {}

        def callee(call):
            frame = ValuationContext(module=mod)._callee_context(call)
            if frame is None:
                return None
            if isinstance(call.target, Function):
                return call.target, ""
            slot = (
                call.target
                if isinstance(call.target, ModuleList)
                else call.target.args[0]
            )
            index = (
                0 if isinstance(call.target, ModuleList) else call.target.args[1].value
            )
            prefix = f"{slot.key}__{index}__" if slot.indexed else f"{slot.key}__"
            return frame.function, prefix

        def needed(fn):
            if id(fn) not in needs:
                names = [p.name for p in fn.params if p.is_const]
                for call in collect_exprs(fn.body):
                    if not isinstance(call, Call) or (item := callee(call)) is None:
                        continue
                    target, prefix = item
                    target_needs = (
                        needed(target)
                        if not prefix
                        else lift(
                            ValuationContext(module=mod, function=fn)._callee_context(call).module
                        )[2][id(target)]
                    )
                    names.extend(prefix + name for name in target_needs)
                needs[id(fn)] = tuple(dict.fromkeys(names))
            return needs[id(fn)]

        def rewrite(fn):
            if id(fn) in functions:
                return functions[id(fn)]
            inputs = tuple(
                Var(p.name, type=p.type) for p in fn.params if not p.is_const
            )
            weights = {name: Var(name, type=specs[name][0]) for name in needed(fn)}
            input_iter = iter(inputs)
            bindings = {
                id(p): weights[p.name] if p.is_const else next(input_iter)
                for p in fn.params
            }

            class Rewrite(BindingSubstitutionCloner):
                def visit_ModuleList(self, slot, ctx=None):
                    return children[slot.key]

                def visit_Call(self, call, ctx=None):
                    args = tuple(self.visit(arg, ctx) for arg in call.args)
                    item = callee(call)
                    if item is None:
                        return replace(call, args=args)
                    target, prefix = item
                    if prefix:
                        child = ValuationContext(module=mod, function=fn)._callee_context(call).module
                        child_needs = lift(child)[2][id(target)]
                        target = self.visit(call.target, ctx)
                    else:
                        args = tuple(
                            arg
                            for param, arg in zip(target.params, args)
                            if not param.is_const
                        )
                        child_needs = needed(target)
                        target = rewrite(target)
                    return replace(
                        call,
                        target=target,
                        args=(*args, *(weights[prefix + name] for name in child_needs)),
                    )

            result = Function.build(
                name=fn.name,
                params=(*inputs, *weights.values()),
                body=Rewrite().visit(fn.body, bindings),
                return_type=fn.return_type,
            )
            functions[id(fn)] = result
            return result

        result = replace(
            mod, functions=tuple(rewrite(fn) for fn in mod.functions), children=children
        )
        cache[id(mod)] = result, specs, needs
        return cache[id(mod)]

    result, specs, needs = lift(root)
    entry = result.entry_function()
    names = needs[id(root.entry_function())]
    weights = set(names)
    params = tuple(
        Var(p.name, type=p.type, is_const=p.name in weights) for p in entry.params
    )
    converters = []
    for name in names:
        _, converter, path = specs[name]
        if converter is None:
            continue
        prefix = "".join(path)
        raw = tuple(
            Var(prefix + p.name, type=p.type, is_const=True) for p in converter.params
        )
        converted = Function.build(
            name=prefix + converter.name,
            params=raw,
            body=BindingSubstitutionCloner().visit(
                converter.body, {id(p): v for p, v in zip(converter.params, raw)}
            ),
            return_type=converter.return_type,
        )
        converters.append((name, converted))
    entry = Function.build(
        name=entry.name,
        params=params,
        body=BindingSubstitutionCloner().visit(
            entry.body, {id(p): v for p, v in zip(entry.params, params)}
        ),
        return_type=entry.return_type,
        converters=tuple(converters),
    )
    return replace(
        result,
        functions=tuple(
            entry if fn.name == entry.name else fn for fn in result.functions
        ),
    )


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
    return {key.replace(".", "__"): value for key, value in out.items()}


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
