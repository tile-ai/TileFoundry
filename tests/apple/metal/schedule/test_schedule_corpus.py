"""Authored Metal HIR checked through the candidate report; scheduled Metal HIR lowered to TIR."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from tests.core._corpus import Case, Problems, case, cases, reload
from tilefoundry import analysis, inspection, utils
from tilefoundry import core as runtime
from tilefoundry.interfaces import evaluator
from tilefoundry.ir import core
from tilefoundry.ir.tir import memory
from tilefoundry.passes import transforms
from tilefoundry.schedule import candidates, finalize

ROOT = Path(__file__).parents[1] / "fixtures"
SCHEDULED = ROOT / "metal_scheduled.py"


@dataclass(frozen=True)
class CandidateState:
    module: object
    entry: object
    candidates: dict


@dataclass(frozen=True)
class MetalCase(Case):
    """One authored Metal HIR fixture and the candidate report it should give."""


def _row(state, label):
    rows = [row for row in state.candidates["lines"] if row["line"].split(":")[0] == label]
    assert len(rows) == 1, (label, state.candidates)
    return rows[0]


def _bindings(fit):
    return dict((p["name"], p["value"]) for p in fit["path"])


def _requires(state, label, instruction, **bindings):
    row = _row(state, label)
    fits = [fit for fit in row["candidates"] if fit["id"] == instruction]
    assert fits, (label, instruction, row)
    assert any(
        all(str(value) == _bindings(fit).get(key) for key, value in bindings.items()) for fit in fits
    ), (label, instruction, bindings, fits)


def _rejects(state, label, instruction, reason):
    row = _row(state, label)
    assert all(fit["id"] != instruction for fit in row["candidates"]), row
    assert any(fit["id"] == instruction and reason in fit["reason"] for fit in row["refused"]), row


@case
class MatMul(MetalCase):
    path: Path = ROOT / "hir_matmul.py"

    def guard(self, state, problems):
        _requires(state, "single", "T.metal.matmul", dtype="f32", groups=2)
        _rejects(state, "integer", "T.metal.matmul", "dtype")
        _rejects(state, "memory", "T.metal.matmul", "no applicable HIR declaration")


@case
class Memory(MetalCase):
    path: Path = ROOT / "hir_memory.py"

    def guard(self, state, problems):
        _requires(state, "load", "T.metal.ldmatrix", dtype="f32", groups=2, row_stride=8)
        _requires(state, "store", "T.metal.stmatrix", dtype="f32", groups=2, row_stride=8)
        _rejects(state, "load", "T.metal.stmatrix", "no applicable HIR declaration")
        _rejects(state, "store", "T.metal.ldmatrix", "no applicable HIR declaration")


@case
class Reduce(MetalCase):
    path: Path = ROOT / "hir_reduce.py"

    def guard(self, state, problems):
        _requires(state, "total", "T.metal.reduce", kind="SUM", dtype="f32", groups=2, width=32)
        _rejects(state, "brain_total", "T.metal.reduce", "dtype")


CASES = cases(ROOT, case_type=MetalCase, pattern="hir_*.py")


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_schedule_case(case: MetalCase) -> None:
    problems = Problems(case)
    module = problems.run("load", case.load)
    checked = problems.run("check", lambda m: analysis.check_program(m, m.entry_function()), module)
    report = problems.run("candidates", lambda pair: candidates(*pair), checked)
    if report is not None:
        problems.run("guard", case.guard, CandidateState(*checked, report), problems)
    problems.dump()


def _lower(module):
    checked = analysis.check_program(module, module.entry_function())
    return finalize(analysis.analyze(*checked, analysis=("memory",)))


def _run_tir(function, *inputs):
    """Run lowered TIR with each issued Op's evaluator, in statement order; give its last argument.

    Core evaluates HIR only. The Metal evaluators write their destination in place.
    """
    torch = utils.torch
    values = {
        id(param): evaluator.TensorValue(runtime.ShardTensor(param.type, data))
        for param, data in zip(function.params, inputs, strict=True)
    }

    def run(statement):
        if isinstance(statement, core.Sequential):
            for inner in statement.body:
                run(inner)
        elif isinstance(statement, core.MeshScope):
            run(statement.body)
        elif isinstance(statement, core.LetStmt) and isinstance(statement.value.target, memory.AllocTensor):
            held = statement.var.type
            data = torch.zeros(held.shape, dtype=evaluator.to_torch_dtype(held.dtype))
            values[id(statement.var)] = evaluator.TensorValue(runtime.ShardTensor(held, data))
        elif isinstance(statement, core.Evaluate):
            args = tuple(values[id(arg)] for arg in statement.args)
            handler = evaluator.registry.get(type(statement.callable), None)
            if handler is None:
                raise TypeError(f"no evaluator for {type(statement.callable).__name__}")
            handler(evaluator.EvaluateContext(op=statement.callable, args=args, result_type=args[0].type))
        else:
            raise TypeError(f"no TIR run for {type(statement).__name__}")

    run(function.body)
    return values[id(function.params[-1])].data


def _same_values(lowered):
    torch = utils.torch
    generator = torch.Generator().manual_seed(0)
    a, b, c = (torch.randint(-4, 5, (2, 8, 8), generator=generator).float() for _ in range(3))
    hir = evaluator.evaluate(Case(SCHEDULED, "MatMul").load(), a.clone(), b.clone(), c.clone())
    tir = _run_tir(lowered.entry_function(), a.clone(), b.clone(), c.clone(), torch.zeros(2, 8, 8))
    assert torch.equal(hir, a @ b + c)
    assert torch.equal(tir, hir)


def test_lowering() -> None:
    problems = Problems(Case(SCHEDULED))
    printed = {}
    for name in ("MatMul", "Reduction"):
        lowered = problems.run(f"lower {name}", lambda name=name: _lower(Case(SCHEDULED, name).load()))
        printed[name] = problems.run(f"print {name}", inspection.as_script, lowered)
        reprinted = problems.run(
            f"reparse {name}", lambda text, name=name: inspection.as_script(reload(text, name)), printed[name]
        )
        problems.expect(f"reprint {name}", lambda again, text: again == text, reprinted, printed[name])
        if name == "MatMul":
            problems.run("evaluate MatMul", _same_values, lowered)
    golden = SCHEDULED.with_suffix(".tir.txt").read_text()
    problems.expect("golden", lambda *texts: "\n".join(texts) == golden, *printed.values())
    problems.dump()


@pytest.mark.parametrize(("name", "message"), [("Unmeshed", "MatMul"), ("CoreReduce", "has no lowering for Reduce")])
def test_lowering_refusal(name: str, message: str) -> None:
    with pytest.raises(transforms.LoweringError, match=message):
        _lower(Case(SCHEDULED, name).load())
