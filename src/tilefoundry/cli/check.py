"""The `check` command: what an implementation produced against its reference."""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import torch

from tilefoundry.cli.source import load_namespace, parse_dims
from tilefoundry.evaluator import evaluate
from tilefoundry.evaluator.value import from_torch_dtype
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.core.module import select as select_module
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.specialize import (
    canonical_specialization_signature,
    display_name,
    residual_dims,
    variant_for,
)
from tilefoundry.ir.types.substitute import substitute_dims
from tilefoundry.runtime import PREDICATES, RuntimeModule
from tilefoundry.runtime.measure import Predicate, check, flatten_outputs
from tilefoundry.runtime.resource import (
    DictResource,
    DrawnResource,
    RuntimeResource,
    SafetensorsResource,
    draw_tensor,
)

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
    parser.add_argument(
        "source", metavar="SOURCE", help="FILE.py:Selector — a Module, a leaf, or a twin"
    )
    parser.add_argument(
        "--expected", action="append", metavar="PATH", help="compare against this file"
    )
    parser.add_argument(
        "--inputs",
        metavar="random|files:A.pt,B.pt",
        help="draw activations, or supply one file per parameter in its declared order",
    )
    parser.add_argument(
        "--weights",
        metavar="random|ckpt:DIR",
        help="draw weights lazily, or read them from a safetensors checkpoint directory",
    )
    parser.add_argument(
        "--out",
        action=Ordered,
        metavar="OUTPUT",
        help=(
            "which output the following --fn apply to: `output` when the function "
            "returns one tensor, `output[0]` `output[1]` ... in return order when it "
            "returns a tuple -- positions, not the names your code gives them"
        ),
    )
    parser.add_argument(
        "--fn", action=Ordered, metavar="F", choices=sorted(PREDICATES), help="a comparison to make"
    )
    for bound in BOUNDS:
        parser.add_argument(f"--{bound}", action=Ordered, type=float, help=argparse.SUPPRESS)
    parser.add_argument(
        "--dim",
        action="append",
        metavar="NAME=V[,V...]",
        help="bind a dimension; several values check dispatch",
    )
    parser.add_argument(
        "--device",
        metavar="DEVICE",
        help=(
            "where inputs and weights are built, and so where the run happens; "
            "defaults to the device the selection's Target declares"
        ),
    )
    parser.add_argument("--json", metavar="PATH", help="write the machine-readable report to PATH")


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
                "cannot be met trains you to ignore FAIL"
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
    """Each combination of stated dimension values, in declaration order."""
    combinations: list[dict[str, int]] = [{}]
    for name, values in dims.items():
        combinations = [{**chosen, name: value} for chosen in combinations for value in values]
    return combinations


@dataclass(frozen=True)
class Selection:
    module: Module
    twin: RuntimeModule | None
    children: tuple[str, ...]
    root: Module
    source: str


@dataclass(frozen=True)
class CheckRequest:
    module: Module
    twin: RuntimeModule | None
    inputs: tuple[Any, ...]
    weights: RuntimeResource
    expected: tuple[Any, ...] | None
    device: str
    expectations: dict[str, tuple[Predicate, ...]]


def _refuse_orchestration(module: Module, method: str) -> None:
    functions = ", ".join(fn.name for fn in module.functions) or "none"
    raise ValueError(
        f"check targets HIR functions, not orchestration method "
        f"{module.name}.{method}; select one of its HIR functions instead: {functions}"
    )


def _module_selection(root: Module, path: Sequence[str]) -> tuple[Module, tuple[str, ...]]:
    node = root
    children: list[str] = []
    for index, segment in enumerate(path):
        child = next((item for item in node.modules if item.name == segment), None)
        if child is not None:
            node = child
            children.append(segment)
            continue
        if index != len(path) - 1:
            raise ValueError(f"selector {'.'.join(path)!r}: {segment!r} is not a child module")
        if segment in node.methods:
            _refuse_orchestration(node, segment)
        node.lookup(segment)
        return select_module(root, ".".join(path)), tuple(children)
    return node, tuple(children)


def select(source: str) -> Selection:
    """Resolve one authored Module or runtime twin without hiding source I/O."""
    namespace, selector = load_namespace(source)
    segments = selector.split(".") if selector else []
    if segments:
        root = namespace.get(segments[0])
        if isinstance(root, type) and issubclass(root, RuntimeModule):
            twin, _top, module, children = _walk_twin(root, segments)
            return Selection(module, twin, children, _top, source)
        if not isinstance(root, Module):
            raise TypeError(f"selector {segments[0]!r} is not a Module or runtime twin")
        chosen, children = _module_selection(root, segments[1:])
        return Selection(chosen, None, children, root, source)

    modules = tuple(value for value in namespace.values() if isinstance(value, Module))
    twins = tuple(
        value for value in namespace.values()
        if isinstance(value, type) and issubclass(value, RuntimeModule) and value is not RuntimeModule
    )
    if len(modules) == 1:
        return Selection(modules[0], None, (), modules[0], source)
    if len(twins) == 1:
        twin, top, module, children = _walk_twin(twins[0], (twins[0].__name__,))
        return Selection(module, twin, children, top, source)
    raise ValueError("source must identify exactly one Module or runtime twin")


def read_inputs(paths: Sequence[str], device: str) -> tuple[Any, ...]:
    return tuple(_read(path, device) for path in paths)


def draw_inputs(module: Module, dims: dict[str, int], seed: int, device: str):
    function = module.entry_function()
    residual_dims(function)
    concrete = replace(
        function,
        params=tuple(
            replace(param, type=substitute_dims(param.type, dims)) for param in function.params
        ),
    )
    generator = torch.Generator(device=device).manual_seed(seed)
    return _random_activations(concrete, generator, device)


def build_resource(
    spec: str | None, module: Module, device: str, generator=None
) -> RuntimeResource:
    if spec is None:
        return DictResource({})
    if spec == "random":
        generator = generator or torch.Generator(device=device).manual_seed(SEED)
        return DrawnResource(module, generator, device)
    if spec.startswith("ckpt:"):
        return SafetensorsResource(spec[5:], device=device)
    raise ValueError("--weights takes random or ckpt:DIR")


def _scope(resource: RuntimeResource, children: Sequence[str]) -> RuntimeResource:
    for child in children:
        resource = resource.subtree(child)
    return resource


def check_concrete(request: CheckRequest):
    loaded = request.module.load(request.weights)
    def reference_run(*args):
        return evaluate(loaded, *args)
    if request.expected is not None:
        expected = request.expected[0] if len(request.expected) == 1 else request.expected
        def expected_run(*_args):
            return expected
        reference = expected_run
    else:
        reference = reference_run if request.twin is not None else None
    if request.twin is None:
        candidate = reference_run
    else:
        request.twin.load(request.weights)
        candidate = getattr(request.twin, request.module.entry_function().name)
    return check(candidate, reference, request.inputs, expect=request.expectations)


def _walk_twin(
    root: type, segments: Sequence[str]
) -> tuple[RuntimeModule, Module, Module, tuple[str, ...]]:
    """Resolve a runtime twin and a dotted path into it."""
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
                "can only be the last segment"
            )
        module = _authored(node)
        if segment in module.methods:
            _refuse_orchestration(module, segment)
        chosen = select_module(module, segment)
        return type(node)(ir=chosen), _authored(top), chosen, tuple(children)
    module = _authored(node)
    if module.methods.get("forward") is not None:
        _refuse_orchestration(module, "forward")
    return node, _authored(top), module, tuple(children)


def _authored(node: RuntimeModule) -> Module:
    """The authored Module a runtime twin stands for."""
    module = node.module
    if module is None:
        raise ValueError(
            f"runtime module {node.name!r} names no authored Module, so there is "
            "nothing to check it against"
        )
    if not isinstance(module, Module):
        raise ValueError(
            f"runtime module {node.name!r}: module must be Module or None, got "
            f"{type(module).__name__}"
        )
    return module


def _device(module: Module) -> str:
    """Choose the execution device declared by a Module, when one is present."""
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


def _random_activations(function: Function, generator, device: str) -> tuple[torch.Tensor, ...]:
    """Draw the non-constant parameters of a concrete function."""
    return tuple(
        draw_tensor(param.type, generator, device)
        for param in function.params
        if not param.is_const
    )


def _read(path: str, device: str):
    """Load one tensor tree from a file and move each leaf to *device*."""
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


def _input_files(paths: Sequence[str], activations: Sequence[Any]) -> list[dict[str, Any]]:
    """Describe the tensor count and structure supplied by each file."""
    files = []
    for path, activation in zip(paths, activations, strict=True):
        leaves = flatten_outputs(activation)
        files.append(
            {
                "path": path,
                "tensor_count": len(leaves),
                "structure": [
                    {
                        "path": position,
                        "dtype": from_torch_dtype(tensor.dtype).name,
                        "shape": list(tensor.shape),
                    }
                    for position, tensor in leaves
                ],
            }
        )
    return files


def _variant(function: Function, dims: dict[str, int]) -> dict[str, Any] | None:
    """Describe the implementation selected for concrete dimensions."""
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


def _shown_dtypes(dtypes: Sequence[str]) -> str:
    """Dtypes as one readable field, including an honest empty declaration."""
    return ", ".join(dtypes) if dtypes else "none"


def _shown_structure(structure) -> str:
    """Render one flattened input tree compactly."""
    shown = [
        f"{item['dtype']}[{', '.join(str(extent) for extent in item['shape'])}]"
        for item in structure
    ]
    return shown[0] if len(shown) == 1 else "(" + ", ".join(shown) + ")"


def _shown_files(files: Sequence[dict[str, Any]]) -> str:
    """Render every input file's count and tensor shape tree."""
    if not files:
        return ""
    descriptions = [
        f"{Path(file['path']).name}: {file['tensor_count']} tensor(s) "
        f"{_shown_structure(file['structure'])}"
        for file in files
    ]
    return "; files " + "; ".join(descriptions)


def _output_dict(output) -> dict[str, Any]:
    """Make a Report output JSON-safe while retaining every measured fact."""
    results = []
    for measured in output.results:
        item = {
            "fn": measured.predicate.name,
            **{bound: getattr(measured.predicate, bound) for bound in measured.predicate.bounds},
            **dict(measured.values),
            "passed": measured.passed,
        }
        if measured.note is not None:
            item["note"] = measured.note
        results.append(item)
    return {
        "path": output.path,
        "shape": list(output.shape),
        "dtype": output.dtype,
        "ref_norm": output.ref_norm,
        "fns": results,
    }


def _failure_warnings(runs: Sequence[dict[str, Any]], input_kind: str | None) -> list[str]:
    """The limits that qualify a failed command-level comparison."""
    if all(run["passed"] for run in runs):
        return []

    warnings = []
    if input_kind == "random":
        warnings.append(
            "--inputs random makes each activation independently. A target that relies on "
            "semantic relationships between activations can differ at ulp scale without either "
            "implementation being wrong. Rerun with --inputs files:... to decide the comparison."
        )
    if any(run["reference"] is not None for run in runs):
        warnings.append(
            "FAIL says the candidate and reference differ, not which side is closer to truth. "
            "The reference may carry its own rounding; check compares only against it. "
            "Establishing accuracy needs an independent high-precision reference, which check "
            "does not run."
        )
    return warnings


def _render(source: str, runs: Sequence[dict[str, Any]], warnings: Sequence[str]) -> str:
    lines = [source]
    for run in runs:
        if run.get("dims"):
            lines += ["", "  " + ", ".join(f"{k}={v}" for k, v in run["dims"].items())]
        lines.append(f"  reference: {run.get('reference', 'none')}")
        activations = run["inputs"]["activations"]
        lines.append(
            f"  inputs:    {activations['source']}; activations actual "
            f"{_shown_dtypes(activations['actual_dtypes'])} (declared "
            f"{_shown_dtypes(activations['declared_dtypes'])})"
            f"{_shown_files(activations.get('files', []))}"
        )
        if run.get("variant") is not None:
            variant = run["variant"]
            ranges = ", ".join(
                f"{item['dim']} in [{item['lo']}, {item['hi']})" for item in variant["ranges"]
            )
            label = variant.get("display_name")
            shown = variant["signature"]
            lines.append(f"  variant:   {shown if label is None else f'{label}  {shown}'}  ({ranges})")
        lines.append("")
        for output in run["outputs"]:
            output = output if isinstance(output, dict) else _output_dict(output)
            shape = ",".join(str(extent) for extent in output["shape"])
            norm = "" if output.get("ref_norm") is None else f"   ref_norm {output['ref_norm']:.6g}"
            lines.append(f"  {output['path']}   {output['dtype']}[{shape}]{norm}")
            for result in output["fns"]:
                bounds = " ".join(
                    f"{bound}={result[bound]:g}"
                    for bound in PREDICATES[result["fn"]].bounds
                )
                values = " ".join(
                    f"{key} {value:g}"
                    for key, value in result.items()
                    if key not in {"fn", "passed", "note", *PREDICATES[result["fn"]].bounds}
                )
                stated = f"{result['fn']}({bounds})" if bounds else result["fn"]
                lines.append(f"    {stated:<34} {values:<26} {'PASS' if result['passed'] else 'FAIL'}")
                if result.get("note"):
                    lines.append(f"      {result['note']}")
        lines.append("")
    passed = all(run["passed"] for run in runs)
    tally = f"  {sum(1 for run in runs if run['passed'])}/{len(runs)}" if len(runs) > 1 else ""
    lines.append(f"{'PASS' if passed else 'FAIL'}{tally}")
    for warning in warnings:
        wrapped = textwrap.wrap(warning, width=74)
        lines += ["", f"  warning: {wrapped[0]}", *(f"           {line}" for line in wrapped[1:])]
    return "\n".join(lines) + "\n"


def run_check(arguments: argparse.Namespace) -> int:
    """Compare one selected implementation against its semantic reference."""
    expect = expectations(getattr(arguments, "comparison", None))
    selection = select(arguments.source)
    stated = parse_dims(arguments.dim) or {}
    if arguments.inputs is None:
        raise ValueError("no inputs stated")
    device = arguments.device or _device(selection.module)
    runs = []
    for dims in _combinations(stated):
        concrete = None
        selected_variant = None
        if arguments.inputs == "random":
            fn = selection.module.entry_function()
            variant_dims = {
                pattern.dim_var
                for variant in fn.variants
                for pattern in variant.specializations
            }
            if variant_dims and variant_dims <= dims.keys():
                selected_variant = _variant(fn, dims)
            concrete = replace(
                fn,
                params=tuple(
                    replace(p, type=substitute_dims(p.type, dims)) for p in fn.params
                ),
            )
            inputs = draw_inputs(selection.module, dims, SEED, device)
        elif arguments.inputs.startswith("files:"):
            inputs = read_inputs(arguments.inputs[6:].split(","), device)
        else:
            raise ValueError("--inputs takes random or files:A.pt,B.pt")
        generator = torch.Generator(device=device).manual_seed(SEED)
        resource = build_resource(arguments.weights, selection.root, device, generator)
        resource = _scope(resource, selection.children)
        expected = None
        if arguments.expected:
            expected_values = read_inputs(arguments.expected, device)
            expected = expected_values
        report = check_concrete(
            CheckRequest(
                selection.module,
                selection.twin,
                inputs,
                resource,
                expected,
                device,
                expect,
            )
        )
        declared = tuple(
            param.type.dtype.name
            for param in (concrete.params if concrete is not None else ())
            if not param.is_const
        )
        if arguments.expected:
            reference = ", ".join(arguments.expected)
        elif selection.twin is not None:
            fn_name = selection.module.entry_function().name
            qualified = ".".join((*selection.children, fn_name))
            if not selection.children:
                qualified = f"{selection.module.name}.{fn_name}"
            reference = f"evaluator on {qualified}"
        else:
            reference = "none — the candidate alone"
        runs.append({
            "passed": report.passed,
            "outputs": [_output_dict(output) for output in report.outputs],
            "dims": dims,
            "reference": reference,
            "variant": selected_variant,
            "inputs": {
                "activations": {
                    "source": f"{arguments.inputs} (seed {SEED})" if arguments.inputs == "random" else arguments.inputs,
                    "actual_dtypes": [
                        from_torch_dtype(tensor.dtype).name
                        for _path, tensor in flatten_outputs(inputs)
                    ],
                    "declared_dtypes": list(declared),
                    "files": _input_files(arguments.inputs[6:].split(","), inputs)
                    if arguments.inputs.startswith("files:") else [],
                },
            },
        })
    warnings = _failure_warnings(runs, arguments.inputs)
    passed = all(run["passed"] for run in runs)
    if arguments.json:
        payload = {"target": arguments.source, "runs": runs, "passed": passed}
        if warnings:
            payload["warnings"] = warnings
        Path(arguments.json).write_text(json.dumps(payload, default=str, indent=2) + "\n", encoding="utf-8")
    else:
        sys.stdout.write(_render(arguments.source, runs, warnings))
    return 0 if passed else 1


__all__ = [
    "Selection",
    "CheckRequest",
    "add_arguments",
    "build_resource",
    "check_concrete",
    "draw_inputs",
    "expectations",
    "guidance",
    "parse_dims",
    "read_inputs",
    "run_check",
    "select",
]
