"""``RuntimeModule`` — a pure run container over already-assembled
``RuntimeFunction`` implementations, nested child modules, and named
weights/states.

Two construction paths:
- compiled: ``tilefoundry.build`` / ``compile`` / ``jit`` → codegen →
  ``LinkedModule`` → ``runtime.loader.load_linked_module`` assembles a
  single-function ``RuntimeModule`` with no ``resource`` / ``weights`` /
  ``states``.
- handwritten: direct construction — a ``RuntimeFunction`` wraps a plain
  torch/triton callable, and a ``RuntimeResource``
  (``tilefoundry.runtime.resource``) supplies weights/states by name.

No evaluator lives here: every ``functions`` entry must already be a
concrete ``RuntimeFunction``.
"""
from __future__ import annotations

from collections.abc import Mapping as ABCMapping
from typing import Any, Callable, Iterator, Mapping

import torch

from tilefoundry.evaluator.value import to_torch_dtype
from tilefoundry.ir.types.tensor_type import TensorType
from tilefoundry.runtime.function import CallableType, KernelInfo, LaunchConfig, RuntimeFunction
from tilefoundry.runtime.resource import RuntimeResource
from tilefoundry.target.base import Device

__all__ = ["RuntimeModule"]


class _LazyMapping(ABCMapping):
    """Read-only mapping whose key set is fixed upfront but whose values
    are computed — and cached — only on first access."""

    def __init__(self, keys: tuple[str, ...], factory: Callable[[str], Any]) -> None:
        self._keys = keys
        self._factory = factory
        self._cache: dict[str, Any] = {}

    def __getitem__(self, key: str) -> Any:
        if key not in self._cache:
            if key not in self._keys:
                raise KeyError(key)
            self._cache[key] = self._factory(key)
        return self._cache[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._keys)

    def __len__(self) -> int:
        return len(self._keys)


def _torch_device_str(device: "Device | None") -> str:
    """Map a ``Device`` to the torch device string used for state allocation.

    A device whose class lives under ``tilefoundry.target.cuda`` maps to
    ``"cuda"``; every other concrete ``Device`` maps to ``"cpu"``. Only
    called where a device is actually required (zero-filling a state with no
    resource-provided initial value, or ``measure.bench``'s timing dispatch)
    — ``None`` is otherwise a perfectly valid ``RuntimeModule`` field.
    """
    if device is None:
        raise ValueError("RuntimeModule: a device is required here, got None")
    if type(device).__module__.startswith("tilefoundry.target.cuda"):
        return "cuda"
    return "cpu"


def _assemble_args(
    fn_type: CallableType,
    weights: Mapping[str, Any],
    states: Mapping[str, Any],
    acts: tuple[Any, ...],
) -> list[Any]:
    """Fill weights/states by name, then decide — from *how many* activation
    arguments were given — how far into ``fn_type.params`` to assemble.

    ``RuntimeModule`` only ever resolves weights/states by name; the
    auto-alloc-vs-pre-alloc calling-convention decision stays with
    ``RuntimeFunction`` (§1.2), so this stops at whichever of its two legal
    boundaries *acts* exactly covers:

    - *acts* fills exactly the unresolved slots of
      ``fn_type.params[:fn_type.input_count]`` (the input segment) →
      assemble only that prefix and stop; ``RuntimeFunction`` then
      auto-allocs the output(s) when ``output_count > 0``.
    - *acts* fills exactly the unresolved slots across *every*
      ``fn_type.params`` entry (input + output segments) → assemble the
      full param list; ``RuntimeFunction`` then pre-allocs (uses the given
      output(s)).

    Any other activation count raises ``ValueError`` naming both legal
    counts. (``output_count == 0`` collapses the two boundaries to the same
    single count — the value-returning handwritten-``fn`` path is
    unaffected.)
    """
    input_count = fn_type.input_count
    resolved: list[tuple[bool, Any]] = []
    for p in fn_type.params:
        if p.name in weights:
            resolved.append((True, weights[p.name]))
        elif p.name in states:
            resolved.append((True, states[p.name]))
        else:
            resolved.append((False, None))

    n_in = sum(1 for bound, _ in resolved[:input_count] if not bound)
    n_full = n_in + fn_type.output_count
    n_acts = len(acts)

    if n_acts == n_in:
        span = resolved[:input_count]
    elif n_acts == n_full:
        span = resolved
    else:
        raise ValueError(
            f"RuntimeModule: {fn_type.name!r} expects {n_in} activation "
            f"argument(s) (weights/states filled by name, output(s) "
            f"auto-alloc'd) or {n_full} (output(s) pre-alloc'd), got {n_acts}"
        )

    acts_iter = iter(acts)
    return [value if bound else next(acts_iter) for bound, value in span]


class RuntimeModule:
    """Live, callable wrapper over a set of ``RuntimeFunction``\\ s, nested
    child ``RuntimeModule``\\ s, and named weights/states resolved from a
    ``RuntimeResource``.
    """

    def __init__(
        self,
        name: str,
        entry: str,
        functions: Mapping[str, RuntimeFunction],
        *,
        resource: RuntimeResource | None = None,
        weight_names: tuple[str, ...] = (),
        state_specs: Mapping[str, TensorType] | None = None,
        modules: tuple["RuntimeModule", ...] = (),
        device: Device | None = None,
        source: str | None = None,
        kernels: tuple[KernelInfo, ...] = (),
        launch_config: LaunchConfig | None = None,
    ) -> None:
        self.name = name
        self.entry = entry
        self.functions: dict[str, RuntimeFunction] = dict(functions)
        self.resource = resource
        self.modules: tuple[RuntimeModule, ...] = tuple(modules)
        self._modules_by_name = {m.name: m for m in self.modules}
        self.source = source
        self.kernels = tuple(kernels)
        self.launch_config = launch_config

        def _load_weight(wname: str) -> torch.Tensor:
            if resource is None:
                raise ValueError(
                    f"RuntimeModule {self.name!r}: no resource to load weight "
                    f"{wname!r} from"
                )
            return resource.load(wname)

        self.weights: Mapping[str, torch.Tensor] = _LazyMapping(tuple(weight_names), _load_weight)

        states: dict[str, torch.Tensor] = {}
        for sname, ty in (state_specs or {}).items():
            loaded = None
            if resource is not None:
                try:
                    loaded = resource.load(sname)
                except KeyError:
                    loaded = None
            if loaded is None:
                loaded = torch.zeros(
                    ty.shape, dtype=to_torch_dtype(ty.dtype), device=_torch_device_str(device)
                )
            states[sname] = loaded
        self.states: dict[str, torch.Tensor] = states

    def __getattr__(self, name: str):
        # Only ever consulted for a name normal attribute lookup missed.
        if name.startswith("_"):
            raise AttributeError(name)
        fn = self.functions.get(name)
        if fn is not None:
            def bound(*acts: Any) -> Any:
                args = _assemble_args(fn.type, self.weights, self.states, acts)
                return fn(*args)
            return bound
        child = self._modules_by_name.get(name)
        if child is not None:
            return child
        raise AttributeError(
            f"RuntimeModule {self.name!r} has no function or child module {name!r}"
        )

    def __call__(self, *acts: Any) -> Any:
        return getattr(self, self.entry)(*acts)

    def walk(self) -> Iterator[tuple[tuple[str, ...], "RuntimeModule"]]:
        """Yield ``(path, node)`` for this node and every nested node,
        walked in parallel with the ``modules`` tree (``()`` for self, then
        each child's path prefixed by its own name)."""
        yield (), self
        for child in self.modules:
            for path, node in child.walk():
                yield (child.name, *path), node
