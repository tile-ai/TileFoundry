"""Encode and validate the root description carried by a compiled library."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from importlib.resources import files

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from tilefoundry import services
from tilefoundry.ir import core, expr, types

from .registry import Role, registry
from .signature import CallableSignature, TensorSignature, UnitIdSignature, called_as

DIM_OPS = frozenset(op.__name__ for op in expr.DIM_OP_TYPES)


def describe_dimension(dimension, path: str):
    """Encode dimension arithmetic using only whitelisted operators."""
    value = expr.static_dim_value(dimension)
    if value is not None:
        return value
    if isinstance(dimension, expr.DimVar):
        return {"name": dimension.name, "lo": dimension.lo, "hi": dimension.hi}
    if isinstance(dimension, expr.Call) and type(dimension.target) in expr.DIM_OP_TYPES:
        return {"op": type(dimension.target).__name__, "args": [describe_dimension(arg, path) for arg in dimension.args]}
    raise ValueError(f"{path}: dimension {dimension!r} is not decoded yet")


def describe_tensor(tensor, path: str) -> dict:
    """Encode host-visible tensor shape and dtype independently of device layout."""
    if not isinstance(tensor, types.TensorType):
        raise ValueError(f"{path}: tensor type is not decoded yet")
    return {
        "dtype": tensor.dtype.name, "shape": [describe_dimension(dim, f"{path}.shape[{axis}]") for axis, dim in enumerate(tensor.shape)],
        **({"sharded": 1} if types.shard_layout_of(tensor.layout) is not None else {}),
    }


def describe_root(module, *, target=None) -> dict:
    """Snapshot the root identity and its declared entry weights."""
    entry = module.entry_function() if module.entry is not None else None
    tgt = module.target or target or getattr(entry, "target", None)
    if tgt is None:
        raise ValueError(f"{module.name}.target: target is not decoded yet")
    root = {
        "name": module.name, "target": tgt.name, "functions": [],
        "weights": {
            param.name: describe_tensor(param.type, f"{module.name}.weights.{param.name}")
            for param in (() if entry is None else entry.params) if param.is_const
        },
        "operations": {},
    }
    if entry is not None:
        root["entry"] = entry.name
        signature = CallableSignature(
            name=entry.name, params=tuple(TensorSignature(param.name, param.type) for param in entry.params),
            output_count=getattr(entry, "output_count", 0),
        )
        convention_target = getattr(entry, "target", tgt)
        if registry.get((type(convention_target), Role.CALLEE, type(entry)), None) is not None:
            signature = called_as(
                entry, entry.name, target=convention_target,
                params=entry.params if isinstance(entry, core.Function) else None,
            )
        root["functions"] = [{"name": entry.name, "signature": describe_signature(signature, entry.params)}]
    return root


def describe_signature(signature, source_params) -> dict:
    """Encode native tensor order and distinguish source inputs from scratch."""
    if signature.trailing or any(not isinstance(param, UnitIdSignature) for param in signature.leading):
        raise ValueError(f"{signature.name}: calling convention is not decoded yet")
    sources = {param.name: param for param in source_params}
    params = []
    for index, param in enumerate(signature.params):
        path = f"{signature.name}.params[{index}]"
        if not isinstance(param, TensorSignature):
            raise ValueError(f"{path}: parameter type is not decoded yet")
        source = sources.get(param.name)
        role = (
            "output" if index >= signature.input_count else
            "scratch" if source is None else "weight" if source.is_const else "input"
        )
        params.append({"name": param.name, "tensor": describe_tensor(param.type, path), "role": role})
    return {
        "params": params, "output_count": signature.output_count,
        "unit_ids": [param.topology for param in signature.leading],
    }


def bind_root(declaration, signature, symbol: str, operations) -> dict:
    """Associate the source entry declaration with the actual native host signature."""
    source = declaration["functions"][0]["signature"]["params"]
    roles = {param["name"]: param["role"] for param in source}
    native = describe_signature(signature, ())
    for param in native["params"][:signature.input_count]:
        param["role"] = roles.get(param["name"], "scratch")
    return {**declaration, "functions": [{"name": declaration["entry"], "symbol": symbol, "signature": native}], "operations": operations}


def decode_dimension(value):
    """Reconstruct a validated host dimension without resolving authored code."""
    if isinstance(value, int):
        return value
    if "op" in value:
        return services.simplify_dim(value["op"][3:].lower(), tuple(map(decode_dimension, value["args"])))
    return expr.DimVar(**value)


def decode_tensor(tensor) -> types.TensorType:
    """Reconstruct only the host tensor type expressed by a validated record."""
    return types.TensorType.from_shape(
        tuple(map(decode_dimension, tensor["shape"])),
        dtype=expr.DType.from_name(tensor["dtype"]), storage=expr.StorageKind.HOST,
    )


def decode_signature(function) -> CallableSignature:
    """Reconstruct the validated tensor and unit-id host ABI without execution IR."""
    signature = function["signature"]
    return CallableSignature(
        name=function["name"],
        params=tuple(TensorSignature(param["name"], decode_tensor(param["tensor"])) for param in signature["params"]),
        output_count=signature["output_count"],
        leading=tuple(UnitIdSignature(f"tilefoundry_{name}_unit_id", name) for name in signature["unit_ids"]),
    )


def standard_value(value, path: str):
    """Copy standard FFI containers into ordinary Python values."""
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError(f"{path}: map keys must be strings")
        return {str(key): standard_value(item, f"{path}.{key}") for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [standard_value(item, f"{path}[{index}]") for index, item in enumerate(value)]
    if type(value) is int and not -(1 << 63) <= value < (1 << 63):
        raise ValueError(f"{path}: integer is outside int64 range")
    if isinstance(value, str):
        return str(value)
    if type(value) is int:
        return value
    raise ValueError(f"{path}: expected a map, array, integer or string")


def fields(value, required: set[str], path: str, optional: frozenset[str] = frozenset()) -> None:
    """Require the exact fields of a schema record."""
    if not isinstance(value, dict) or not required <= value.keys() or value.keys() - required - optional:
        raise ValueError(f"{path}: expected fields {sorted(required)}, optional {sorted(optional)}")


def text_value(value, path: str, *, symbol: bool = False) -> None:
    """Require a nonempty string or a C identifier."""
    if not isinstance(value, str) or not value or (symbol and not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", value)):
        raise ValueError(f"{path}: expected {'a C identifier' if symbol else 'a nonempty string'}")


def array_value(value, path: str) -> None:
    """Require an array after standard-value conversion."""
    if not isinstance(value, list):
        raise ValueError(f"{path}: expected an array")


def validate_tensor(tensor, path: str, dimensions: dict, *, expression: bool = False) -> None:
    """Check the tensor whitelist and consistent bounds for named dimensions."""
    fields(tensor, {"dtype", "shape"}, path, frozenset({"sharded"}))
    if "sharded" in tensor and (type(tensor["sharded"]) is not int or tensor["sharded"] != 1):
        raise ValueError(f"{path}.sharded: expected 1")
    if not isinstance(tensor["dtype"], str) or tensor["dtype"] not in expr.dtype_members():
        raise ValueError(f"{path}.dtype: unknown dtype {tensor['dtype']!r}")
    array_value(tensor["shape"], f"{path}.shape")
    for axis, dimension in enumerate(tensor["shape"]):
        location = f"{path}.shape[{axis}]"
        if type(dimension) is int:
            if dimension < 0 and not expression:
                raise ValueError(f"{location}: extent must be nonnegative")
            continue
        if isinstance(dimension, dict) and "op" in dimension:
            fields(dimension, {"op", "args"}, location)
            if not isinstance(dimension["op"], str) or dimension["op"] not in DIM_OPS or not isinstance(dimension["args"], list) or len(dimension["args"]) != 2:
                raise ValueError(f"{location}: expected a known binary dimension operator")
            validate_tensor({"dtype": tensor["dtype"], "shape": dimension["args"]}, location, dimensions, expression=True)
            continue
        fields(dimension, {"name", "lo", "hi"}, location)
        text_value(dimension["name"], f"{location}.name")
        lo, hi = dimension["lo"], dimension["hi"]
        if type(lo) is not int or type(hi) is not int or not 0 <= lo <= hi:
            raise ValueError(f"{location}: expected 0 <= lo <= hi")
        bounds = (lo, hi)
        if dimensions.setdefault(dimension["name"], bounds) != bounds:
            raise ValueError(f"{location}: inconsistent bounds for {dimension['name']!r}")


def validate_description(description) -> dict:
    """Reject malformed identities, tensor declarations, roles and symbols."""
    description = standard_value(description, "manifest")
    fields(description, {"format", "root"}, "manifest")
    if description["format"] != 1:
        raise ValueError("manifest.format: expected format 1")
    root = description["root"]
    fields(root, {"name", "target", "functions", "weights", "operations"}, "root", frozenset({"entry"}))
    for key in ("name", "target"):
        text_value(root[key], f"root.{key}")
    array_value(root["functions"], "root.functions")
    if len(root["functions"]) != (1 if "entry" in root else 0):
        raise ValueError("root.functions: expected one entry function, or none without root.entry")
    if "entry" in root:
        text_value(root["entry"], "root.entry")
    dimensions, symbols = {}, {"tilefoundry_manifest"}
    if not isinstance(root["weights"], dict) or not isinstance(root["operations"], dict):
        raise ValueError("root.weights/root.operations: expected maps")
    for name, tensor in root["weights"].items():
        text_value(name, "root.weights.name")
        validate_tensor(tensor, f"root.weights.{name}", dimensions)
    for index, function in enumerate(root["functions"]):
        path = f"root.functions[{index}]"
        fields(function, {"name", "symbol", "signature"}, path)
        if function["name"] != root["entry"]:
            raise ValueError(f"{path}.name: must equal root.entry")
        sig = function["signature"]
        fields(sig, {"params", "output_count", "unit_ids"}, f"{path}.signature")
        array_value(sig["params"], f"{path}.signature.params")
        count = sig["output_count"]
        if type(count) is not int or not 0 <= count <= len(sig["params"]):
            raise ValueError(f"{path}.signature.output_count: outside parameter range")
        names = set()
        for offset, param in enumerate(sig["params"]):
            location = f"{path}.signature.params[{offset}]"
            fields(param, {"name", "tensor", "role"}, location)
            text_value(param["name"], f"{location}.name")
            if param["name"] in names:
                raise ValueError(f"{location}.name: duplicate parameter")
            names.add(param["name"])
            role = param["role"]
            if role not in ("input", "weight", "scratch", "output"):
                raise ValueError(f"{location}.role: unknown role")
            if (role == "output") != (offset >= len(sig["params"]) - count):
                raise ValueError(f"{location}.role: outputs must be trailing")
            validate_tensor(param["tensor"], f"{location}.tensor", dimensions)
            weight = root["weights"].get(param["name"])
            if (role == "weight" and weight is None) or (weight is not None and (role != "weight" or weight != param["tensor"])):
                raise ValueError(f"{location}: inconsistent weight declaration")
        array_value(sig["unit_ids"], f"{path}.signature.unit_ids")
        for topology in sig["unit_ids"]:
            text_value(topology, f"{path}.signature.unit_ids")
        if len(set(sig["unit_ids"])) != len(sig["unit_ids"]):
            raise ValueError(f"{path}.signature.unit_ids: duplicate topology")
    exports = [(f"root.functions[{index}].symbol", fn["symbol"]) for index, fn in enumerate(root["functions"])]
    exports.extend((f"root.operations.{name}", symbol) for name, symbol in root["operations"].items())
    for path, symbol in exports:
        text_value(symbol, path, symbol=True)
        if symbol in symbols:
            raise ValueError(f"{path}: duplicate symbol {symbol!r}")
        symbols.add(symbol)
    for name in root["operations"]:
        text_value(name, "root.operations.name")
    return description


def cpp_value(value) -> str:
    """Render validated standard values with explicit FFI container types."""
    if isinstance(value, dict):
        pairs = ("{" + cpp_value(key) + ", " + cpp_value(item) + "}" for key, item in value.items())
        return "tvm::ffi::Map<tvm::ffi::String, tvm::ffi::Any>{" + ", ".join(pairs) + "}"
    if isinstance(value, list):
        return "tvm::ffi::Array<tvm::ffi::Any>{" + ", ".join(cpp_value(item) for item in value) + "}"
    if isinstance(value, str):
        encoded = value.encode("utf-8")
        literal = "".join(chr(byte) if 32 <= byte < 127 and byte not in (34, 92) else f"\\{byte:03o}" for byte in encoded)
        return f'tvm::ffi::String("{literal}", {len(encoded)})'
    return f"int64_t{{{value}}}"


def render_description(description) -> str:
    """Render the manifest translation unit from validated standard values."""
    env = Environment(
        loader=FileSystemLoader(str(files("tilefoundry.codegen").joinpath("templates"))),
        undefined=StrictUndefined, keep_trailing_newline=True,
    )
    return env.get_template("artifact.cpp.j2").render(value=cpp_value(validate_description(description)))


__all__ = ["bind_root", "decode_signature", "decode_tensor", "describe_root", "describe_signature", "render_description", "validate_description"]
