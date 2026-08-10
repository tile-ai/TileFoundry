"""The `check` command: what an implementation produced against its reference."""

from __future__ import annotations

import argparse
import inspect
import json
import sys
import textwrap
from pathlib import Path
from typing import Any, Sequence

import torch

from tilefoundry.cli.source import load_namespace, parse_dims, select_ir, suggested_extents
from tilefoundry.evaluator.value import to_torch_dtype
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.hir.function import Function, canonical_specialization_signature
from tilefoundry.ir.hir.specialize import (
    dim_vars_reached,
    display_name,
    specialize_concretely,
    variant_for,
)
from tilefoundry.runtime import PREDICATES, RuntimeModule, SafetensorsResource, check
from tilefoundry.runtime.measure import Predicate

SEED = 0


BOUNDS: tuple[str, ...] = tuple(
    dict.fromkeys(bound for predicate in PREDICATES.values() for bound in predicate.bounds)
)


class Ordered(argparse.Action):
    """Record `--out`, `--fn` and the bounds in the order they were written.

    argparse collects repeated options into one flat list per option, which
    loses which `--fn` a bound belongs to and which `--out` a `--fn` is under.
    """

    def __call__(self, parser, namespace, values, option_string=None):
        stated = getattr(namespace, "comparison", None)
        if stated is None:
            stated = []
            setattr(namespace, "comparison", stated)
        stated.append((self.dest, values))


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Declare `check`'s own arguments, bounds included, off the registry."""
    parser.add_argument("source", metavar="SOURCE", help="FILE.py:Selector — a Module, a leaf, or a twin")
    parser.add_argument(
        "--inputs",
        choices=("random", "real"),
        help="how activations and weights are made: random, or real from --ckpt",
    )
    parser.add_argument(
        "--input",
        action="append",
        metavar="PATH",
        help="an activation file; repeat per input, in the parameter's declared order",
    )
    parser.add_argument("--ckpt", metavar="DIR", help="a prepared checkpoint directory")
    parser.add_argument(
        "--expected", action="append", metavar="PATH", help="compare against this file"
    )
    parser.add_argument(
        "--out",
        action=Ordered,
        metavar="OUTPUT",
        help=(
            "which output the following --fn apply to: `output` when the function "
            "returns one tensor, `output[0]` `output[1]` ... in return order when "
            "it returns a tuple -- positions, not the names your code gives them"
        ),
    )
    parser.add_argument(
        "--fn", action=Ordered, metavar="F", choices=sorted(PREDICATES), help="a comparison to make"
    )
    for bound in BOUNDS:
        parser.add_argument(f"--{bound}", action=Ordered, type=float, help=argparse.SUPPRESS)
    parser.add_argument(
        "--dim", action="append", metavar="NAME=V[,V...]", help="bind a dimension; several values check dispatch"
    )
    parser.add_argument("--json", action="store_true", help="machine-readable report")


def guidance() -> str:
    """The comparison functions and their bounds, from the registry itself."""
    lines = [
        "comparison — every output needs at least one --fn. There is no default:",
        "a default bound that cannot be met trains you to ignore FAIL.",
        "",
    ]
    width = max(len(name) for name in PREDICATES)
    for name in sorted(PREDICATES):
        predicate = PREDICATES[name]
        bounds = " ".join(f"--{bound} X" for bound in predicate.bounds)
        lines.append(f"  --fn {name:<{width}} {bounds:<18} {predicate.guidance}")
    lines += [
        "",
        "deriving rtol for a bf16 result: bf16 has 8 explicit mantissa bits, so one",
        "round-to-nearest is at most 2^-9 = 1.95e-3 per element. Count the serial",
        "f32->bf16 landings in your own implementation -- one fused kernel accumulating",
        "in f32 lands once, three chained kernels land three times -- and take",
        "rtol ~ 2^-9 * sqrt(n) for the longest serial chain. Parallel branches do not",
        "accumulate. atol covers the near-zero elements, where rtol * |b| degenerates.",
    ]
    return "\n".join(lines)


def expectations(stated: Sequence[tuple[str, Any]] | None) -> dict[str, tuple[Predicate, ...]]:
    """The `--out` / `--fn` / bound sequence as the predicates each output states."""
    if not stated:
        raise ValueError(
            "no comparison requested. Every output needs --out PATH and at least "
            "one --fn; see `tilefoundry check --help` for the functions there are"
        )
    grouped: dict[str, list[tuple[str, dict[str, float]]]] = {}
    current: str | None = None
    predicate: tuple[str, dict[str, float]] | None = None
    for flag, value in stated:
        if flag == "out":
            if value in grouped:
                raise ValueError(f"--out {value!r} was given twice; state its functions together")
            current = value
            grouped[current] = []
            predicate = None
            continue
        if flag == "fn":
            if current is None:
                raise ValueError(f"--fn {value} came before any --out; name the output first")
            predicate = (value, {})
            grouped[current].append(predicate)
            continue
        if predicate is None:
            raise ValueError(f"--{flag} came before any --fn; name the function it bounds first")
        name, bounds = predicate
        if flag not in PREDICATES[name].bounds:
            raise ValueError(
                f"--fn {name} takes {list(PREDICATES[name].bounds) or 'no bounds'}, not --{flag}"
            )
        if flag in bounds:
            raise ValueError(f"--{flag} was given twice for one --fn {name}")
        bounds[flag] = value

    built: dict[str, tuple[Predicate, ...]] = {}
    for path, predicates in grouped.items():
        if not predicates:
            raise ValueError(
                f"--out {path!r} states no --fn. There is no default bound: one that "
                f"cannot be met trains you to ignore FAIL"
            )
        made = []
        for name, bounds in predicates:
            missing = [bound for bound in PREDICATES[name].bounds if bound not in bounds]
            if missing:
                raise ValueError(
                    f"--fn {name} needs {['--' + bound for bound in missing]}; a bound has no default"
                )
            made.append(PREDICATES[name](**bounds))
        built[path] = tuple(made)
    return built


def _combinations(dims: dict[str, tuple[int, ...]]) -> list[dict[str, int]]:
    """Each combination of the stated values, in the order they were stated."""
    combinations: list[dict[str, int]] = [{}]
    for name, values in dims.items():
        combinations = [{**chosen, name: value} for chosen in combinations for value in values]
    return combinations


class Target:
    """What a `SOURCE` resolved to: something to run, and the Module behind it.

    `children` is the child-module part of the selector, which the resource is
    scoped by. `module` is that node in the tree, not the re-entered copy
    `ir.core.module.select` returns for a terminal function.
    """

    def __init__(self, source: str) -> None:
        path_text, _, selector = source.partition(":")
        self.path = Path(path_text).expanduser().resolve()
        namespace, _ = load_namespace(source)
        segments = selector.split(".") if selector else []
        first = namespace.get(segments[0]) if segments else None

        self.twin: RuntimeModule | None = None
        if isinstance(first, type) and issubclass(first, RuntimeModule):
            (
                self.twin, self.top, self.module, self.children, self.function_name
            ) = _walk_twin(first, segments)
        else:
            self.top = select_ir(namespace, segments[0] if segments else None)
            self.module, self.children, self.function_name = _walk_ir(self.top, segments[1:])
        self.selector = selector

    @property
    def function(self) -> Function | None:
        """The one HIR function this target runs, when it runs exactly one."""
        if self.function_name is None:
            return None
        return self.module.lookup(self.function_name)


def _walk_ir(top: Module, path: Sequence[str]) -> tuple[Module, tuple[str, ...], str | None]:
    """*path* below *top*, as (the Module, the child names walked, the function named).

    *path* below *top*, as (the Module, the child names walked, the function
    named). A non-child segment must be last and must name one of the reached
    Module's functions.
    """
    reached = top
    children: list[str] = []
    for index, name in enumerate(path):
        below = {child.name: child for child in reached.modules}
        if name in below:
            reached = below[name]
            children.append(name)
            continue
        if index != len(path) - 1:
            raise ValueError(
                f"selector {'.'.join(path)!r}: Module {reached.name!r} has no child "
                f"module {name!r}"
            )
        reached.lookup(name)
        return reached, tuple(children), name
    if reached.methods.get("forward") is not None:
        return reached, tuple(children), None
    return reached, tuple(children), reached.entry


def _walk_twin(
    root: type, segments: Sequence[str]
) -> tuple[RuntimeModule, Module, Module, tuple[str, ...], str | None]:
    """A twin class and a dotted path into it.

    A twin class and a dotted path into it, as (node, the top Module, the
    node's Module, the child names walked, the function named).
    """
    top = root()
    node = top
    children: list[str] = []
    for index, segment in enumerate(segments[1:]):
        try:
            found = getattr(node, segment)
        except AttributeError as error:
            raise ValueError(f"selector {'.'.join(segments)!r}: {error}") from None
        if isinstance(found, RuntimeModule):
            node = found
            children.append(segment)
            continue
        if index != len(segments) - 2:
            raise ValueError(
                f"selector {'.'.join(segments)!r}: {segment!r} is a function, so it "
                f"can only be the last segment"
            )
        return node, _authored(top), _authored(node), tuple(children), segment
    return node, _authored(top), _authored(node), tuple(children), None


def _authored(node: RuntimeModule) -> Module:
    """The Module a twin stands for, refusing one that stands for nothing."""
    module = node.module
    if module is None:
        raise ValueError(
            f"runtime module {node.name!r} names no authored Module, so there is "
            f"nothing to check it against. `check` takes a @runtime_module twin, "
            f"or the authored Module with --expected"
        )
    if not isinstance(module, Module):
        raise ValueError(
            f"runtime module {node.name!r}: module must be Module or None, got "
            f"{type(module).__name__}"
        )
    return module


def _device(module: Module) -> str:
    """Where to build the tensors.

    A declared CUDA Target is a requirement -- its kernels run nowhere else. A
    selection that declares no Target is a Module being compared through the
    evaluator, which runs wherever there is a device.
    """
    try:
        target = module.resolve_target()
    except Exception:
        return "cuda" if torch.cuda.is_available() else "cpu"
    if not type(target).__module__.startswith("tilefoundry.target.cuda"):
        return "cpu"
    if not torch.cuda.is_available():
        raise ValueError(
            f"{module.name!r} declares {target.name}, and this machine has no CUDA device"
        )
    return "cuda"


def _draw(shape: Sequence[int], dtype, generator: torch.Generator, device: str) -> torch.Tensor:
    """One tensor of *dtype*, drawn from *generator*."""
    torch_dtype = to_torch_dtype(dtype)
    if torch_dtype.is_floating_point:
        drawn = torch.randn(tuple(shape), generator=generator, device=device)
        return drawn.to(torch_dtype)
    if torch_dtype == torch.bool:
        return torch.randint(0, 2, tuple(shape), generator=generator, device=device).to(torch_dtype)
    return torch.randint(0, 8, tuple(shape), generator=generator, device=device).to(torch_dtype)


def _extents(type_) -> tuple[int, ...]:
    """A concrete shape, refusing one still stated as a range."""
    shape = []
    for extent in type_.shape:
        if not isinstance(extent, int):
            raise ValueError(
                f"shape {type_.shape} still states {extent} as a range; bind it with --dim"
            )
        shape.append(extent)
    return tuple(shape)


def _random_activations(function: Function, generator, device: str) -> tuple[torch.Tensor, ...]:
    """One seeded draw of everything the function takes that is not a weight."""
    return tuple(
        _draw(_extents(param.type), param.type.dtype, generator, device)
        for param in function.params
        if not param.is_const
    )


class RandomWeights:
    """Each weight drawn the first time it is asked for, from its declared type.

    Drawn rather than pre-built so a leaf never materialises the tensors its
    siblings declare, and cached so the two sides of a comparison are handed the
    same draw rather than two draws of the same shape.
    """

    def __init__(self, module: Module, generator, device: str, drawn=None, prefix: str = "") -> None:
        self._module = module
        self._generator = generator
        self._device = device
        self._drawn: dict[str, torch.Tensor] = {} if drawn is None else drawn
        self._prefix = prefix

    def load(self, name: str) -> torch.Tensor:
        key = f"{self._prefix}{name}"
        if key not in self._drawn:
            declared = self._module.weights[name]
            self._drawn[key] = _draw(
                _extents(declared), declared.dtype, self._generator, self._device
            )
        return self._drawn[key]

    def load_group(self, name: str):
        return None

    def subtree(self, seg: str) -> "RandomWeights":
        for child in self._module.modules:
            if child.name == seg:
                return RandomWeights(
                    child, self._generator, self._device, self._drawn, f"{self._prefix}{seg}."
                )
        raise KeyError(seg)


def _read(path: str, device: str):
    """One parameter tree from a file, moving every tensor leaf to *device*."""
    found = Path(path).expanduser()
    if found.suffix == ".npy":
        import numpy  # noqa: PLC0415 -- only this path needs it

        loaded = torch.from_numpy(numpy.load(found))
    else:
        loaded = torch.load(found, map_location=device, weights_only=True)

    def visit(value, position: str):
        if isinstance(value, torch.Tensor):
            return value.to(device)
        if isinstance(value, (list, tuple)):
            return tuple(visit(item, f"{position}[{index}]") for index, item in enumerate(value))
        raise ValueError(
            f"{position}: expected a tensor or nested tuple/list of tensors, "
            f"got {type(value).__name__}"
        )

    return visit(loaded, path)


def _tensor_leaves(value):
    """Every tensor in one activation tree, in its written order."""
    if isinstance(value, torch.Tensor):
        yield value
        return
    for item in value:
        yield from _tensor_leaves(item)


def _tensor_structure(value):
    """The JSON-safe shape tree that an activation file supplied."""
    if isinstance(value, torch.Tensor):
        return {"dtype": str(value.dtype), "shape": list(value.shape)}
    return [_tensor_structure(item) for item in value]


def _dtype_names(values: Sequence[Any]) -> list[str]:
    """Actual torch dtypes of every tensor across the supplied values."""
    return [str(tensor.dtype) for value in values for tensor in _tensor_leaves(value)]


def _input_files(paths: Sequence[str], activations: Sequence[Any]) -> list[dict[str, Any]]:
    """The leaf count and structure that each stated activation file supplied."""
    return [
        {
            "path": path,
            "tensor_count": sum(1 for _ in _tensor_leaves(activation)),
            "structure": _tensor_structure(activation),
        }
        for path, activation in zip(paths, activations, strict=True)
    ]


def _weights_needed(module: Module) -> tuple[str, ...]:
    """Every weight the selected Module declares, which is what a run binds."""
    return tuple(module.weights)


def _resource(target: Target, generator, device: str, ckpt: str | None):
    """Where both sides read their weights.

    Where both sides read their weights: one seeded draw, or the checkpoint,
    rooted at the top-level Module and scoped by *target*'s child names.
    """
    resource = (
        RandomWeights(target.top, generator, device)
        if ckpt is None
        else SafetensorsResource(ckpt, device=device)
    )
    for name in target.children:
        resource = resource.subtree(name)
    return resource


def _variant(function: Function, dims: dict[str, int]) -> dict[str, Any] | None:
    """Which implementation this size dispatches to, and over what range."""
    if not function.variants:
        return None
    chosen = variant_for(function, dims)
    ranges = [
        {"dim": pattern.dim_var, "lo": pattern.lo, "hi": pattern.hi}
        for pattern in chosen.specializations
    ]
    label = display_name(chosen)
    return {
        **({} if label is None else {"display_name": label}),
        "signature": canonical_specialization_signature(chosen.specializations),
        "ranges": ranges,
    }


def _pin(function: Function, stated: dict[str, int]) -> tuple[dict[str, int], list[dict[str, Any]]]:
    """Every dimension this function states as a range, bound to one extent.

    A dimension the caller named keeps that extent; one nobody named is pinned to
    the first value of its declared range and reported, because a run happened at
    one size whether or not anybody chose it.
    """
    declared = dim_vars_reached(function)
    bound: dict[str, int] = {}
    unstated: list[dict[str, Any]] = []
    for name, dim_var in declared.items():
        if name in stated:
            bound[name] = stated[name]
            continue
        bound[name] = dim_var.lo
        unstated.append(
            {
                "dim": name,
                "pinned": dim_var.lo,
                "lo": dim_var.lo,
                "hi": dim_var.hi,
                "spread": suggested_extents(dim_var.lo, dim_var.hi),
            }
        )
    unknown = sorted(set(stated) - set(declared))
    if unknown:
        raise ValueError(
            f"--dim {unknown} name no dimension of {function.name!r}; it states "
            f"{sorted(declared)}"
        )
    return bound, unstated


def _sides(target: Target, resource, expected: Sequence[str] | None, device: str):
    """The two callables to compare, and how to say what the reference was.

    The reference is the selected Module loaded whole, then its function.
    """
    name = target.function_name
    loaded = target.module.load(resource)
    evaluator = getattr(loaded, name) if name else loaded.forward

    if target.twin is not None:
        target.twin.load(resource)
        candidate = getattr(target.twin, name) if name else target.twin.forward
    else:
        candidate = evaluator

    if expected:
        tensors = tuple(_read(path, device) for path in expected)
        one = tensors[0] if len(tensors) == 1 else tensors
        return candidate, (lambda *_: one), ", ".join(expected), loaded.constants

    if target.twin is None:


        return candidate, None, None, loaded.constants
    authored = target.module.name if name is None else f"{target.module.name}.{name}"
    return candidate, evaluator, f"evaluator on {authored}", loaded.constants


def _orchestration_parameter_names(target: Target) -> tuple[str, ...]:
    """The activations the selected orchestration method actually accepts."""
    method = target.twin.forward if target.twin is not None else target.module.methods["forward"]
    return tuple(name for name in inspect.signature(method).parameters if name != "self")


def _one_run(
    target: Target,
    stated: dict[str, int],
    expect: dict[str, tuple[Predicate, ...]],
    arguments: argparse.Namespace,
    device: str,
) -> dict[str, Any]:
    """One comparison at one set of extents, as the facts both outputs carry."""
    function = target.function
    pinned: dict[str, int] = {}
    unstated: list[dict[str, Any]] = []
    concrete = function
    if function is not None:
        pinned, unstated = _pin(function, stated)
        if pinned:
            concrete = specialize_concretely(function, pinned)
    elif stated:
        raise ValueError(
            f"--dim was given, but {target.module.name!r} runs an orchestration "
            f"method rather than one function, so there is no signature to bind"
        )

    needed = _weights_needed(target.module)
    if needed and arguments.ckpt is None and arguments.inputs != "random":
        raise ValueError(
            f"{target.module.name!r} needs weights {list(needed)} and no source was "
            f"given; draw them with --inputs random or read them with --ckpt DIR"
        )

    generator = torch.Generator(device=device).manual_seed(SEED)
    activations: tuple[Any, ...]
    if arguments.input:
        activations = tuple(_read(path, device) for path in arguments.input)
        provided = f"{len(activations)} file(s): {', '.join(arguments.input)}"
    elif arguments.inputs is None:
        raise ValueError(
            "no inputs stated. Give exactly one form and no default: --inputs random, "
            "--inputs real --ckpt DIR, or --input=PATH per activation"
        )
    else:
        if concrete is None:
            names = _orchestration_parameter_names(target)
            raise ValueError(
                f"--inputs {arguments.inputs} cannot make activations for "
                f"{target.module.name!r}: it runs an orchestration method whose "
                f"parameters have no declared shapes or dtypes. It takes {len(names)} "
                f"activation parameters in order: {', '.join(names)}. Give one "
                "--input=PATH per parameter; each file holds one tensor or a nested "
                "tuple/list of tensors, for example torch.save((...), \"mixer_args.pt\")"
            )
        activations = _random_activations(concrete, generator, device)
        provided = f"random, seed {SEED}"

    resource = _resource(target, generator, device, arguments.ckpt)
    if not needed:
        weights_from = "none declared"
    else:
        weights_from = "the checkpoint" if arguments.ckpt else f"random, seed {SEED}"

    candidate, reference, reference_label, loaded_weights = _sides(
        target, resource, arguments.expected, device
    )
    report = check(candidate, reference, activations, expect=expect)
    return {
        "dims": dict(stated),
        "pinned": unstated,
        "variant": None if concrete is None else _variant(function, pinned),
        "inputs": {
            "activations": {
                "source": provided,
                "actual_dtypes": _dtype_names(activations),
                "declared_dtypes": (
                    []
                    if concrete is None
                    else [
                        parameter.type.dtype.name
                        for parameter in concrete.params
                        if not parameter.is_const
                    ]
                ),
                "files": _input_files(arguments.input, activations) if arguments.input else [],
            },
            "weights": {
                "source": weights_from,
                "actual_dtypes": _dtype_names(tuple(loaded_weights.values())),
                "declared_dtypes": [type_.dtype.name for type_ in target.module.weights.values()],
            },
        },
        "reference": reference_label,
        "outputs": [
            {
                "path": output.path,
                "shape": list(output.shape),
                "dtype": output.dtype,
                **({} if output.ref_norm is None else {"ref_norm": output.ref_norm}),
                "fns": [
                    {
                        "fn": result.predicate.name,
                        **{
                            bound: getattr(result.predicate, bound)
                            for bound in result.predicate.bounds
                        },
                        **result.values,
                        "passed": result.passed,
                        **({} if result.note is None else {"note": result.note}),
                    }
                    for result in output.results
                ],
            }
            for output in report.outputs
        ],
        "passed": report.passed,
    }


def _shown_dtypes(dtypes: Sequence[str]) -> str:
    """Dtypes as one readable field, including an honest empty declaration."""
    return ", ".join(dtypes) if dtypes else "none"


def _shown_structure(structure) -> str:
    """One recursively-loaded input tree, compactly enough for the inputs line."""
    if isinstance(structure, dict):
        return f"{structure['dtype']}[{', '.join(str(extent) for extent in structure['shape'])}]"
    return "(" + ", ".join(_shown_structure(item) for item in structure) + ")"


def _shown_files(files: Sequence[dict[str, Any]]) -> str:
    """Every file's count plus its tensor shape tree, for the text report."""
    if not files:
        return ""
    descriptions = [
        f"{Path(file['path']).name}: {file['tensor_count']} tensor(s) "
        f"{_shown_structure(file['structure'])}"
        for file in files
    ]
    return "; files " + "; ".join(descriptions)


def _failure_warnings(runs: Sequence[dict[str, Any]], input_kind: str | None) -> list[str]:
    """The limits that qualify a failed command-level comparison."""
    if all(run["passed"] for run in runs):
        return []

    warnings = []
    if input_kind == "random":
        warnings.append(
            "--inputs random makes each activation independently. A target that relies on "
            "semantic relationships between activations can differ at ulp scale without either "
            "implementation being wrong. Rerun with --inputs real to decide the comparison."
        )
    if any(run["reference"] is not None for run in runs):
        warnings.append(
            "FAIL says the candidate and reference differ, not which side is closer to truth. "
            "The reference may carry its own rounding; check compares only against it. "
            "Establishing accuracy needs an independent high-precision reference, which check "
            "does not run."
        )
    return warnings


def _render(
    target: Target,
    runs: list[dict[str, Any]],
    warnings: Sequence[str],
) -> str:
    """The runs as a person reads them: what ran, what it measured, the verdict."""
    where = f"{target.path.name}:{target.selector}" if target.selector else str(target.path.name)
    lines = [where]
    for run in runs:
        if run["dims"]:
            lines.append("")
            lines.append(f"  {', '.join(f'{k}={v}' for k, v in run['dims'].items())}")
        lines.append(f"  reference: {run['reference'] or 'none — the candidate alone'}")
        activations = run["inputs"]["activations"]
        weights = run["inputs"]["weights"]
        lines.append(
            f"  inputs:    {activations['source']}; activations actual "
            f"{_shown_dtypes(activations['actual_dtypes'])} (declared "
            f"{_shown_dtypes(activations['declared_dtypes'])}); weights "
            f"{weights['source']} actual {_shown_dtypes(weights['actual_dtypes'])} "
            f"(declared {_shown_dtypes(weights['declared_dtypes'])})"
            f"{_shown_files(activations['files'])}"
        )
        if run["variant"] is not None:
            ranges = ", ".join(
                f"{r['dim']} in [{r['lo']}, {r['hi']})" for r in run["variant"]["ranges"]
            )
            named = run["variant"].get("display_name")
            shown = run["variant"]["signature"]
            lines.append(
                f"  variant:   {shown if named is None else f'{named}  {shown}'}  ({ranges})"
            )
        for pinned in run["pinned"]:
            lines.append(
                f"  note: {pinned['dim']} is a range [{pinned['lo']}, {pinned['hi']}) that "
                f"nothing bound; this run pinned it to {pinned['pinned']}."
            )
            spread = ",".join(str(value) for value in pinned["spread"])
            lines.append(f"        tilefoundry check ... --dim {pinned['dim']}={spread}")
            lines.append(
                "        to make the size a declared variant instead of a pin, see "
                "`tilefoundry spec parser 1.1`"
            )
        lines.append("")
        for output in run["outputs"]:
            shape = ",".join(str(extent) for extent in output["shape"])
            norm = "" if "ref_norm" not in output else f"   ref_norm {output['ref_norm']:.6g}"
            lines.append(f"  {output['path']}   {output['dtype']}[{shape}]{norm}")
            for measured in output["fns"]:
                bounds = " ".join(
                    f"{bound}={getattr(PREDICATES[measured['fn']], bound, None) or measured[bound]:g}"
                    for bound in PREDICATES[measured["fn"]].bounds
                )
                values = " ".join(
                    f"{key} {value:g}"
                    for key, value in measured.items()
                    if key not in ("fn", "passed", "note", *PREDICATES[measured["fn"]].bounds)
                )
                verdict = "PASS" if measured["passed"] else "FAIL"
                stated = f"{measured['fn']}({bounds})" if bounds else measured["fn"]
                lines.append(f"    {stated:<34} {values:<26} {verdict}")
                if "note" in measured:
                    lines.append(f"      {measured['note']}")
        lines.append("")
    passed = all(run["passed"] for run in runs)
    tally = f"  {sum(1 for run in runs if run['passed'])}/{len(runs)}" if len(runs) > 1 else ""
    lines.append(f"{'PASS' if passed else 'FAIL'}{tally}")
    for warning in warnings:
        wrapped = textwrap.wrap(warning, width=74)
        lines += ["", f"  warning: {wrapped[0]}", *(f"           {line}" for line in wrapped[1:])]
    return "\n".join(lines) + "\n"


def run_check(arguments: argparse.Namespace) -> int:
    """Compare one target against its reference and report every output."""
    expect = expectations(getattr(arguments, "comparison", None))
    stated = parse_dims(arguments.dim) or {}
    if arguments.input and arguments.inputs is not None:
        raise ValueError(
            f"--input names the activations and --inputs {arguments.inputs} makes them; "
            f"give exactly one form. Weights alongside --input come from --ckpt DIR"
        )
    if arguments.ckpt and arguments.inputs == "random":
        raise ValueError("--inputs random draws its own weights; --ckpt would not be read")
    if arguments.inputs == "real" and not arguments.ckpt:
        raise ValueError("--inputs real reads real weights, so it needs --ckpt DIR")

    target = Target(arguments.source)
    device = _device(target.module)
    runs = [
        _one_run(target, combination, expect, arguments, device)
        for combination in _combinations(stated)
    ]
    warnings = _failure_warnings(runs, arguments.inputs)

    if arguments.json:
        payload = {
            "target": f"{target.path.name}:{target.selector}" if target.selector else target.path.name,
            "runs": runs,
            "passed": all(run["passed"] for run in runs),
        }
        if warnings:
            payload["warnings"] = warnings
        sys.stdout.write(json.dumps(payload, indent=2, sort_keys=False) + "\n")
    else:
        sys.stdout.write(_render(target, runs, warnings))
    return 0 if all(run["passed"] for run in runs) else 1


__all__ = ["add_arguments", "expectations", "guidance", "parse_dims", "run_check"]
