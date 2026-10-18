"""Golden answers at the analysis boundary consumed by HIR lowering."""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections import defaultdict
from pathlib import Path

import pytest

from tilefoundry.analysis import analyze
from tilefoundry.analysis.access import AliasKind, SSAValue
from tilefoundry.analysis.check import check_program
from tilefoundry.ir.core import Call, Var, value_labels
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.loop_region import LoopRegion
from tilefoundry.ir.hir.tensor.insert_slice import InsertSlice
from tilefoundry.ir.types import ComposedLayout, DType, Layout, StorageKind, Swizzle, TensorType
from tilefoundry.ir.types.utils import i64_const
from tilefoundry.ir.visitor import collect_exprs
from tilefoundry.passes.transforms.convert_hir_to_tir import (
    LoweringError,
    StoragePlan,
    TirBuilder,
    TirScope,
)
from tilefoundry.visitor_registry.access_relation import leaves_of
from tilefoundry.visitor_registry.call_frame import CallFrame, CallFrameVisitor, ValuationVisitor
from tilefoundry.visitor_registry.contexts import TypeInferContext

FIXTURES = Path(__file__).parents[1] / "fixtures" / "schedule"
HIR = tuple(sorted((FIXTURES / "hir").glob("*.py")))
GOLDEN = FIXTURES / "storage"


def storage_answers(source: Path) -> list[dict]:
    """Read each reachable definition through the same context as lowering.

    The lexical index includes region parameters omitted by an operand walk;
    reachable operands also supply constants and forwarded tuple leaves.
    """
    spec = importlib.util.spec_from_file_location(source.stem, source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    authored = next(value for value in vars(module).values() if type(value).__name__ == "Module")
    result = analyze(*check_program(authored, authored.entry_function()), analysis=("memory",))
    frames = tuple(CallFrameVisitor.walk_frames(CallFrame(result.module, result.function)))
    definitions, names = [], {}
    for index, frame in enumerate(frames):
        function = frame.function
        locations = ValuationVisitor.index_contexts(
            function, TypeInferContext(frame=frame, owner=function)
        )
        expressions = {id(function): function, **{id(param): param for param in function.params}}
        for scope in locations.values():
            if scope.call is not None:
                expressions[id(scope.call)] = scope.call
            if scope.owner is not None:
                expressions[id(scope.owner)] = scope.owner
                for param in getattr(scope.owner, "params", ()):
                    expressions[id(param)] = param
                if isinstance(scope.owner, LoopRegion):
                    expressions[id(scope.owner.induction_var)] = scope.owner.induction_var
        expressions.update((id(expr), expr) for expr in collect_exprs(function.body))
        values = tuple(expressions.values())
        prefix = f"f{index}:{function.name}"
        names.update(
            (id(expr), f"{prefix}/{label}")
            for expr, label in zip(values, value_labels(values), strict=True)
        )
        definitions.append((prefix, function, frame, values))

    def label(leaf):
        return f"{names[id(leaf.expr)]}[{leaf.index}]"

    answers = []
    for prefix, function, frame, values in definitions:
        context = StoragePlan(function, frame)
        queries = defaultdict(list)
        for expr in values:
            if isinstance(expr, Call) and isinstance(expr.target, InsertSlice):
                queries[SSAValue(expr)].append((AliasKind.INSERT_SLICE, SSAValue(expr.args[0])))
                queries[SSAValue(expr.args[1])].append((AliasKind.WINDOW, SSAValue(expr)))
            if isinstance(expr, LoopRegion):
                for param, initial, yielded in zip(
                    expr.params, expr.args, expr.yield_values, strict=False
                ):
                    for index in range(len(leaves_of(param.type))):
                        phi = SSAValue(param, index)
                        queries[phi].append((AliasKind.CARRY_INIT, SSAValue(initial, index)))
                        queries[SSAValue(yielded, index)].append((AliasKind.CARRY_YIELD, phi))
        rows = []
        for expr in values:
            for index in range(len(leaves_of(expr.type))):
                leaf = SSAValue(expr, index)
                if leaf not in context.answers:
                    continue
                partner = context.answers[leaf].partner
                rows.append(
                    {
                        "leaf": label(leaf),
                        "present": leaf in context.answers,
                        "partner": f"output[{partner}]"
                        if isinstance(partner, int)
                        else label(partner)
                        if partner is not None
                        else None,
                        "kind": [
                            kind.name
                            for kind in (alias.kind for alias in context.answers[leaf].aliases)
                        ],
                        "tied": [
                            {
                                "kind": kind.name,
                                "partner": label(other),
                                "value": any(
                                    alias.kind is kind and alias.target == other
                                    for alias in context.answers[leaf].aliases
                                ),
                            }
                            for kind, other in queries[leaf]
                        ],
                        "offsets": list(context.answers[leaf].offsets),
                        "refusal": context.answers[leaf].refusal,
                    }
                )
        answers.append({"function": prefix, "leaves": rows})
    return answers


@pytest.mark.parametrize("source", HIR, ids=lambda path: path.stem)
def test_storage_answers(source: Path) -> None:
    assert storage_answers(source) == json.loads((GOLDEN / f"{source.stem}.json").read_text())


@pytest.mark.parametrize("instruction", (False, True), ids=("storage", "instruction"))
@pytest.mark.parametrize(
    ("bits", "starts", "phase"), ((2, (8, 16), 16), (3, (16, 0), 0), (3, (0, 16), 16))
)
def test_swizzled_window_phase_is_relative_to_its_pointer(instruction, bits, starts, phase):
    source_type = TensorType(
        (32, 64),
        DType.bf16,
        ComposedLayout(Swizzle(bits, 4, 3), 0, Layout((32, 64), (64, 1))),
        StorageKind.SMEM,
    )
    source = Var("source", type=source_type)
    function = Function.build(name="window", params=(source,), body=source, return_type=source_type)
    plan = StoragePlan(function)
    desired = TensorType(
        (16, 16),
        DType.bf16,
        ComposedLayout(Swizzle(bits, 4, 3), 0, Layout((16, 16), (64, 1))),
        StorageKind.SMEM,
    )
    coordinates = tuple(i64_const(start) for start in starts)
    actual_starts, actual = plan.window(source, desired, coordinates, instruction=instruction)
    assert actual_starts == coordinates
    assert actual.layout.offset == phase
    assert actual.layout.inner == source_type.layout.inner


def test_loop_induction_cannot_escape_the_builder_region():
    builder = TirBuilder([TirScope()])
    induction = Var("k", type=TensorType.umat_scalar())
    bounds = (induction, i64_const(0), i64_const(4), i64_const(1))
    with pytest.raises(LoweringError, match="value 'k' escapes region mesh"):
        with builder.region(loop=bounds) as region:
            region.result = induction


def main() -> None:
    """Regenerate only the storage context answer snapshots."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", required=True)
    parser.parse_args()
    GOLDEN.mkdir(exist_ok=True)
    for source in HIR:
        (GOLDEN / f"{source.stem}.json").write_text(
            json.dumps(storage_answers(source), indent=2) + "\n"
        )
        print(source.stem, flush=True)


if __name__ == "__main__":
    main()
