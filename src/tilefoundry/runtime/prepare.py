"""Offline weight preparation of a Module: the implementation of ``services.prepare``.

Converters are HIR bodies on this offline path, so preparation runs the
reference evaluator; it lives above both the Module and the evaluator.
"""

from __future__ import annotations

import json
from pathlib import Path

from tilefoundry import services, utils
from tilefoundry.interfaces import evaluator
from tilefoundry.ir import core
from tilefoundry.ir import utils as irutils

from .resource import DictResource

safetensors_torch = utils.LazyModule("safetensors.torch")

__all__ = ["prepare", "prepare_into"]


def prepare(module, raw, out_dir: str, *, device: str = "cpu") -> None:
    """Prepare canonical weights and write the safetensors checkpoint."""
    staged: dict[str, object] = {}
    prepare_into(module, raw, "", staged, device, target=module.target)
    flat = {name: value.detach().contiguous().cpu() for name, value in staged.items()}

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    shard = "model-00001-of-00001.safetensors"
    safetensors_torch.save_file(flat, str(out / shard))
    (out / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: shard for name in flat}})
    )


def prepare_into(
    module, raw, prefix: str, flat: dict, device: str, *,
    target=None, topologies=(), path: str = "",
) -> "core.LoadedModule":
    """Stage this node's canonical weights, children before their owner.

    Converters are HIR bodies on this offline path, making the local
    evaluator import the one intentional Module-to-evaluator execution dependency.
    """
    target = module.target if module.target is not None else target
    topologies = module.topologies if module.topologies is not None else topologies
    children = {}
    for key, slot in module.children.items():
        base = raw.subtree(key)
        child_path = f"{path}.{key}" if path else key
        if isinstance(slot, tuple):
            children[key] = tuple(
                prepare_into(child,
                    base.subtree(str(i)), f"{prefix}{key}.{i}.", flat, device,
                    target=target, topologies=topologies, path=f"{child_path}.{i}",
                )
                for i, child in enumerate(slot)
            )
        else:
            children[key] = prepare_into(slot,
                base, f"{prefix}{key}.", flat, device,
                target=target, topologies=topologies, path=child_path,
            )
    converter_map: dict[str, core.ModuleFunction] = {}
    for fn in module.functions:
        for weight_name, conv in getattr(fn, "converters", ()):
            prior = converter_map.get(weight_name)
            if prior is not None and prior is not conv:
                raise ValueError(
                    f"Module {module.name!r}: weight {weight_name!r} has "
                    f"more than one registered converter"
                )
            converter_map[weight_name] = conv

    def fetch(name):
        parts = raw.load_group(name)
        value = utils.torch.stack(parts) if parts is not None else raw.load(name)
        return value.to(device)

    reading = core.LoadedModule(
        module=module, resource=DictResource(flat, prefix=prefix), children=children,
        target=target, topologies=topologies, path=path,
    )
    for weight_name, decl_type in module.weights.items():
        conv = converter_map.get(weight_name)
        key = prefix + weight_name
        if conv is None:
            value = fetch(weight_name)
        else:
            value = evaluator.run_bound(
                conv, [fetch(param.name) for param in conv.params],
                device=device, reading=reading,
            )
        actual = evaluator.tensor_type_of(value, like=decl_type)
        if actual.shape != decl_type.shape:
            source = (
                f"converter for weight {key!r}"
                if conv is not None
                else f"raw weight {key!r}"
            )
            raise ValueError(
                f"Module {module.name!r}: {source} has shape "
                f"{actual.shape}, declared {decl_type.shape}"
            )
        if not irutils.types_compatible(decl_type, actual):
            source = (
                f"converter for weight {key!r}"
                if conv is not None
                else f"raw weight {key!r}"
            )
            raise ValueError(
                f"Module {module.name!r}: {source} has dtype {value.dtype}, declared "
                f"{decl_type.dtype}; the way out is a weight converter on the model, "
                "not a flag on the read side"
            )
        flat[key] = value
    return reading


services.registry.register(services.prepare, prepare)
