"""Provide runtime twins of IR modules.

``CompiledModule`` is the compiled-path variant bound by the loader. See
[runtime §1.1](docs/spec/runtime.md#11-runtimemodulepy).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Callable

from tilefoundry import codegen, target, utils
from tilefoundry import core as tensor_core
from tilefoundry.interfaces import evaluator
from tilefoundry.ir import core, types

from .function import CompiledFunction
from .resource import Resource

__all__ = ["CompiledModule", "Module", "validate_names"]

tvm_ffi = utils.LazyModule("tvm_ffi")


class Module:
    """Base runtime module: explicit child registration + recursive load.

    Subclasses override ``forward`` (the step) and ``load`` (own weights,
    ending with ``super().load``).
    """

    name: str
    """Mirrors the IR ``Module`` node name."""

    entry: str | None
    """Mirrors the IR ``Module`` entry."""

    children: dict[str, Module | tuple[Module, ...]]
    """Instance slots, registered explicitly in ``__init__``."""

    module: "core.Module | None" = None
    """The authored ``Module`` this twin stands for, or ``None`` when there is
    no such thing to name -- a compiled entry or a hand-written subclass."""

    def __init__(
        self, name: str, entry: str | None = None,
        children: dict[str, Module | tuple[Module, ...]] | None = None,
    ) -> None:
        self.name = name
        self.entry = entry
        self.children = {} if children is None else dict(children)

    def forward(self, *args):
        raise NotImplementedError(f"runtime.Module {self.name!r}: subclass must implement forward()")

    def __call__(self, *args):
        return self.forward(*args)

    def load(self, resource: Resource, *, placement: "types.Placement | None" = None) -> None:
        """Recurse ``load`` into each slot under its key and optional index.

        Weight
        values are read lazily by each runtime function on first use, and
        *placement* -- which unit of the mesh this process is -- reaches
        every child unchanged, because one process is one unit throughout
        the tree it loaded.
        """
        for key, child in self.children.items():
            base = resource.subtree(key)
            if isinstance(child, tuple):
                for index, instance in enumerate(child):
                    instance.load(base.subtree(str(index)), placement=placement)
            else:
                child.load(base, placement=placement)


def validate_names(names, implementation, path: str) -> None:
    """Reject authored attributes that shadow explicitly declared runtime state."""
    reserved = {
        name for base in implementation.__mro__
        for name in (*vars(base), *vars(base).get("__annotations__", {}))
    }
    for name in names:
        if name in reserved:
            raise ValueError(f"{path}.{name}: name is reserved by {implementation.__name__}")


def exported_function(library, path, symbol):
    """Resolve a local function and require its typed metadata."""
    try:
        function = library.get_function(symbol, query_imports=False)
    except AttributeError as error:
        raise RuntimeError(f"library {path}: missing symbol {symbol!r}; recompile the library") from error
    try:
        metadata = library.get_function_metadata(symbol, query_imports=False)
        if metadata is None or "type_schema" not in metadata:
            raise RuntimeError(f"library {path}: missing metadata for symbol {symbol!r}")
        schema = json.loads(metadata["type_schema"])
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"library {path}: invalid metadata for symbol {symbol!r}") from error
    return function, schema


class CompiledModule(Module):
    """A library-owned root entry with explicit resource and call lifetime."""

    library: object
    library_path: Path
    type: codegen.CallableSignature | None
    fn: Callable | None
    weights: dict[str, types.TensorType]
    operations: dict[str, str]
    resource: Resource | None
    placement: types.Placement | None
    bound: dict[str, object]
    closed: bool

    def __init__(self, library, library_path, description) -> None:
        root = description["root"]
        super().__init__(name=root["name"], entry=root.get("entry"))
        self.library = library
        self.library_path = Path(library_path).resolve()
        self.weights = {name: codegen.artifact.decode_tensor(tensor) for name, tensor in root["weights"].items()}
        self.operations = root["operations"]
        self.resource = None
        self.placement = None
        self.bound = {}
        self.closed = False
        self.type = None
        self.fn = None
        if root["functions"]:
            function, = root["functions"]
            self.type = codegen.artifact.decode_signature(function)
            self.fn, schema = exported_function(library, self.library_path, function["symbol"])
            expected = {"type": "ffi.Function", "args": [{"type": name} for name in self.entry_schema()]}
            if schema != expected:
                raise RuntimeError(f"library {self.library_path}: symbol {function['symbol']!r} metadata does not match signature")
            self.__dict__[self.entry] = CompiledFunction(self, self.entry, self.type)
        for symbol in self.operations.values():
            exported_function(library, self.library_path, symbol)

    @classmethod
    def from_library(cls, path) -> CompiledModule:
        """Restore one independent root using only its compiled library."""
        path = Path(path).resolve()
        library = tvm_ffi.load_module(str(path), keep_module_alive=False)
        manifest, _ = exported_function(library, path, "tilefoundry_manifest")
        try:
            description = codegen.artifact.validate_description(manifest())
            root = description["root"]
            provider = target.registered().get(root["target"])
            if provider is None:
                raise RuntimeError(f"library {path}: {root['name']}: target provider {root['target']!r} is not decoded yet")
            implementation = target.bindings.get((provider, CompiledModule), CompiledModule)
            validate_names((function["name"] for function in root["functions"]), implementation, root["name"])
            return implementation(library, path, description)
        except (TypeError, ValueError) as error:
            raise RuntimeError(f"library {path}: {error}") from error

    def entry_schema(self) -> tuple[str, ...]:
        """State the typed FFI return and parameter types of this node's entry."""
        return ("None", *("int" for _ in self.type.leading), *("ffi.Tensor" for _ in self.type.params))

    def export_library(self, path) -> None:
        """Copy the compiled unit without persisting resource or placement state."""
        destination = Path(path)
        if destination.exists() and destination.samefile(self.library_path):
            return
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(destination)
        with self.library_path.open("rb") as source, destination.open("xb") as output:
            try:
                shutil.copyfileobj(source, output)
            except BaseException:
                destination.unlink()
                raise

    def load(self, resource: Resource, *, placement: "types.Placement | None" = None) -> None:
        """Replace this root's resource and release previously bound weights."""
        if self.closed:
            raise RuntimeError(f"{self.name}: compiled module is closed")
        self.bound.clear()
        self.resource = resource
        self.placement = placement
        super().load(resource, placement=placement)

    def close(self) -> None:
        """Release resource bindings and disable calls; repeated closure is safe."""
        self.bound.clear()
        self.resource = None
        self.placement = None
        self.fn = None
        self.closed = True

    def operation(self, name: str):
        """Resolve one implemented lifecycle operation from this library."""
        symbol = self.operations.get(name)
        if symbol is None:
            raise RuntimeError(f"library {self.library_path}: {self.name}.{name} is not decoded yet")
        return exported_function(self.library, self.library_path, symbol)[0]

    def unit_ids(self) -> tuple:
        """Read the leading topology ids from this root's placement."""
        if not self.type.leading:
            return ()
        topologies = tuple(param.topology for param in self.type.leading)
        if self.placement is None:
            raise ValueError(f"{self.name}: needs the ids of {topologies}; pass placement= to load()")
        return tuple(int(self.placement.ids[topology]) for topology in topologies)

    def weight_of(self, name: str):
        """Read, validate and retain one static root weight without conversion."""
        if name in self.bound:
            return self.bound[name]
        if self.resource is None:
            raise KeyError(f"{self.name}: weight {name!r} is not bound; call load(resource) first")
        declared = self.weights[name]
        if any(not isinstance(dimension, int) for dimension in declared.shape):
            raise RuntimeError(f"{self.name}: weight {name!r} with dynamic shape is not decoded yet")
        try:
            value = self.resource.load(name)
        except KeyError as error:
            raise KeyError(f"{self.name}: missing weight {name!r}") from error
        if isinstance(value, tensor_core.ShardTensor):
            raise RuntimeError(f"{self.name}: sharded weight {name!r} is not decoded yet")
        if not isinstance(value, utils.torch.Tensor):
            raise ValueError(f"{self.name}: weight {name!r} must be a tensor")
        try:
            dtype = evaluator.to_torch_dtype(declared.dtype)
        except (evaluator.EvalError, AttributeError) as error:
            raise RuntimeError(f"{self.name}: weight {name!r} dtype {declared.dtype} is not decoded yet") from error
        if tuple(value.shape) != declared.shape or value.dtype != dtype:
            raise ValueError(f"{self.name}: weight {name!r} has shape/dtype {tuple(value.shape)}/{value.dtype}, declared {declared.shape}/{declared.dtype}")
        self.bound[name] = value
        return value

    def forward(self, *args):
        if self.closed:
            raise RuntimeError(f"{self.name}: compiled module is closed")
        if self.type is None:
            raise RuntimeError(f"{self.name}: no entry; functions=[]")
        leading = self.unit_ids()
        n_in = sum(param.name not in self.weights for param in self.type.input_params)
        n_all = n_in + self.type.output_count
        if len(args) not in (n_in, n_all):
            raise TypeError(f"{self.name}.{self.entry}: expected {n_in} inputs (auto-alloc) or {n_all} inputs+outputs (pre-alloc), got {len(args)}")
        inputs = iter(args[:n_in])
        try:
            native = tuple(self.weight_of(param.name) if param.name in self.weights else next(inputs) for param in self.type.input_params)
        except BaseException:
            self.bound.clear()
            raise
        outs = args[n_in:] if len(args) == n_all else self.alloc_outputs(native)
        self.fn(*leading, *native, *outs)
        return outs[0] if len(outs) == 1 else tuple(outs)

    def alloc_outputs(self, args) -> tuple:
        """Allocate static trailing outputs on the input tensor's device."""
        if not self.type.output_count:
            return ()
        if any(not isinstance(dimension, int) for param in self.type.output_params for dimension in param.type.shape):
            raise RuntimeError(f"{self.name}: dynamic output allocation is not decoded yet")
        device = next((arg.device for arg in args if isinstance(arg, utils.torch.Tensor)), None)
        if device is None:
            raise TypeError(f"{self.name}: cannot infer device for auto-alloc; no torch.Tensor in inputs")
        return tuple(utils.torch.empty(param.type.shape, dtype=evaluator.to_torch_dtype(param.type.dtype), device=device) for param in self.type.output_params)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, entry={self.entry!r}, closed={self.closed})"
