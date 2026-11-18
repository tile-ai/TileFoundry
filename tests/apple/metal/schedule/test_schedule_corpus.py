"""Metal HIR offered to the candidate report; hand-scheduled Metal IR evaluated and lowered.

A fixture under ``hir/`` is checked through the candidate report. A fixture under
``scheduled/`` is loaded, type-checked, evaluated by core's HIR evaluator, lowered,
printed as TIR and reloaded; lowered TIR is not evaluated.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from tests.core._corpus import Case, Problems, case, cases, reload
from tilefoundry import analysis, inspection, utils
from tilefoundry import core as runtime
from tilefoundry.interfaces import evaluator
from tilefoundry.ir import core, expr, hir, types
from tilefoundry.passes import transforms
from tilefoundry.schedule import candidates, finalize
from tilefoundry_apple.metal.ir import tir

ROOT = Path(__file__).parents[1] / "fixtures"

MATRICES = tuple((utils.torch.arange(128.0) % mod).reshape(2, 8, 8) for mod in (5, 3, 7))
LANE = utils.torch.arange(32, dtype=utils.torch.int32).repeat(2, 1)
WIDE = LANE.long().index_fill(1, utils.torch.tensor([7]), 0xFFFFFFF0).to(utils.torch.uint32)


def u16(value):
    """A rank-zero u16 operand: a shuffle selector or delta."""
    return utils.torch.tensor(value, dtype=utils.torch.uint16)


@dataclass(frozen=True)
class MetalCase(Case):
    """One Metal fixture; a field names the error its stage must raise.

    ``refusal`` is the load-time VerifyError, ``evaluate_refusal`` the EvalError of
    core's HIR evaluator and ``finalize_refusal`` the LoweringError.
    """

    refusal: str | None = None
    evaluate_refusal: str | None = None
    finalize_refusal: str | None = None

    def inputs(self) -> tuple | None:
        """The entry's torch inputs, or None when the case is not evaluated."""
        return None


@dataclass(frozen=True)
class CandidateState:
    module: object
    entry: object
    candidates: dict


@dataclass(frozen=True)
class ScheduledState:
    module: core.Module | None
    result: object
    lowered: core.Module | None


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


class _Issued(core.StmtVisitor[None]):
    def visit_Evaluate(self, stmt, found):
        found.append(stmt)


def _attribute(op, declared):
    value = getattr(op, declared.name)
    if declared.has_default and value == declared.default:
        return None
    return f"{declared.name}={getattr(value, 'name', value)}"


def issued(lowered: core.Module) -> list[str]:
    """Each statement the lowered entry issues: its Op, operands and non-default attributes.

    An operand is a parameter's name or ``t<i>``, a temporary numbered by its first use.
    """
    function = lowered.entry_function()
    statements: list[core.Evaluate] = []
    _Issued().visit(function, statements)
    params = {param.name for param in function.params}
    temporaries: dict[str, str] = {}

    def operand(arg):
        return arg.name if arg.name in params else temporaries.setdefault(arg.name, f"t{len(temporaries)}")

    described = []
    for stmt in statements:
        op = stmt.callable
        attributes = (_attribute(op, declared) for declared in expr.signature_of(type(op)) if declared.kind == "attribute")
        fields = [*map(operand, stmt.args), *filter(None, attributes)]
        described.append(f"{expr.get_registered_name(type(op))}({', '.join(fields)})")
    return described


def _hir(case, problems):
    module = problems.run("load", case.load)
    checked = problems.run("check", lambda m: analysis.check_program(m, m.entry_function()), module)
    report = problems.run("candidates", lambda pair: candidates(*pair), checked)
    if report is not None:
        problems.run("guard", case.guard, CandidateState(*checked, report), problems)


def _refused(case):
    with pytest.raises(expr.VerifyError, match=case.refusal):
        case.load()


def _evaluate(case, module):
    inputs = tuple(value.clone() for value in case.inputs())
    if case.evaluate_refusal is None:
        return evaluator.evaluate(module, *inputs)
    with pytest.raises(evaluator.EvalError) as refused:
        evaluator.evaluate(module, *inputs)
    assert case.evaluate_refusal in str(refused.value)
    return None


def _finalize(case, checked):
    analyzed = analysis.analyze(*checked, analysis=("memory",))
    if case.finalize_refusal is None:
        return finalize(analyzed)
    with pytest.raises(transforms.LoweringError) as refused:
        finalize(analyzed)
    assert case.finalize_refusal in str(refused.value)
    return None


def _scheduled(case, problems):
    if case.refusal is not None:
        problems.run("load refusal", _refused, case)
        return
    module = problems.run("load", case.load)
    checked = problems.run("check", lambda m: analysis.check_program(m, m.entry_function()), module)
    result = None
    if case.inputs() is not None:
        result = problems.run("evaluate", _evaluate, case, module)
    lowered = problems.run("finalize", _finalize, case, checked)
    if case.finalize_refusal is None:
        printed = problems.run("print TIR", inspection.as_script, lowered)
        lowered = problems.run("reload TIR", lambda text: reload(text, lowered.name), printed)
    problems.run("guard", case.guard, ScheduledState(module, result, lowered), problems)


@case
class MatMulCandidates(MetalCase):
    path: Path = ROOT / "hir" / "matmul.py"

    def guard(self, state, problems):
        _requires(state, "single", "T.metal.matmul", dtype="f32", groups=2)
        _rejects(state, "integer", "T.metal.matmul", "dtype")
        _rejects(state, "memory", "T.metal.matmul", "no applicable HIR declaration")


@case
class MemoryCandidates(MetalCase):
    path: Path = ROOT / "hir" / "memory.py"

    def guard(self, state, problems):
        _requires(state, "load", "T.metal.ldmatrix", dtype="f32", groups=2, row_stride=8)
        _requires(state, "store", "T.metal.stmatrix", dtype="f32", groups=2, row_stride=8)
        _rejects(state, "load", "T.metal.stmatrix", "no applicable HIR declaration")
        _rejects(state, "store", "T.metal.ldmatrix", "no applicable HIR declaration")


@case
class ReduceCandidates(MetalCase):
    path: Path = ROOT / "hir" / "reduce.py"

    def guard(self, state, problems):
        _requires(state, "total", "T.metal.reduce", kind="SUM", dtype="f32", groups=2, width=32)
        _rejects(state, "brain_total", "T.metal.reduce", "dtype")


@case
class LdMatrix(MetalCase):
    """The loaded fragment, stored back as is, gives lane t the elements the measured map names."""

    path: Path = ROOT / "scheduled" / "ldmatrix.py"

    def inputs(self):
        return (utils.torch.arange(128.0).reshape(2, 8, 8),)

    def guard(self, state, problems):
        (a,) = self.inputs()
        assert utils.torch.equal(state.result, a.mT)
        held = next(
            node.type for node in core.collect_exprs(state.module.entry_function().body)
            if isinstance(node, expr.Call) and isinstance(node.target, hir.schedule.ScheduleOp)
            and isinstance(node.target.op, tir.LdMatrix)
        )
        fragment = runtime.ShardTensor(held, state.result)
        seen = []
        for thread in range(64):
            group, lane = divmod(thread, 32)
            row = ((lane >> 1) & 3) + ((lane >> 4) & 1) * 4
            col = (lane & 1) * 2 + ((lane >> 3) & 1) * 4
            local = fragment.to_local(types.Placement({"thread": thread})).flatten().tolist()
            assert local == state.result[group, row, col:col + 2].tolist(), thread
            seen.extend(local)
        assert sorted(seen) == list(range(128))
        assert issued(state.lowered) == ["T.metal.ldmatrix(t0, a, transpose=True)", "T.metal.stmatrix(out, t0)"]


@case
class StMatrix(MetalCase):
    path: Path = ROOT / "scheduled" / "stmatrix.py"

    def inputs(self):
        return (utils.torch.arange(128.0).reshape(2, 8, 8),)

    def guard(self, state, problems):
        assert utils.torch.equal(state.result, self.inputs()[0].mT)
        assert issued(state.lowered) == ["T.metal.ldmatrix(t0, a)", "T.metal.stmatrix(out, t0, transpose=True)"]


@case
class MatMul(MetalCase):
    path: Path = ROOT / "scheduled" / "matmul.py"

    def inputs(self):
        return MATRICES[:2]

    def guard(self, state, problems):
        a, b = MATRICES[:2]
        assert utils.torch.equal(state.result, a @ b)
        assert issued(state.lowered) == [
            "T.metal.ldmatrix(t0, a)", "T.metal.ldmatrix(t1, b)", "T.metal.matmul(t2, t0, t1)",
            "T.metal.stmatrix(out, t2)",
        ]


@case
class Mma(MetalCase):
    """A fresh destination and the accumulator itself as destination give the same a @ b + c."""

    path: Path = ROOT / "scheduled" / "mma.py"

    def inputs(self):
        return MATRICES

    def guard(self, state, problems):
        a, b, c = MATRICES
        fresh, in_place = state.result
        assert utils.torch.equal(fresh, a @ b + c)
        assert utils.torch.equal(in_place, a @ b + c)
        assert issued(state.lowered) == [
            "T.metal.ldmatrix(t0, a)", "T.metal.ldmatrix(t1, b)", "T.metal.ldmatrix(t2, c)",
            "T.metal.mma(t3, t0, t1, t2)", "T.metal.stmatrix(out, t3)",
            "T.metal.mma(t2, t0, t1, t2)", "T.metal.stmatrix(out_1, t2)",
        ]


@case
class Reduce(MetalCase):
    path: Path = ROOT / "scheduled" / "reduce.py"

    def inputs(self):
        return LANE, WIDE, utils.torch.tensor([[1.0], [2.0]])

    def guard(self, state, problems):
        total, largest, wide_largest, resolved = state.result
        assert total.tolist() == [[496]] * 2
        assert largest.tolist() == [[31]] * 2
        assert wide_largest.tolist() == [[0xFFFFFFF0]] * 2
        assert resolved.tolist() == [[1.0], [2.0]]
        assert issued(state.lowered) == [
            "T.metal.reduce(out, x, dtype=i32)", "T.metal.reduce(out_1, x, kind=MAX, dtype=i32)",
            "T.metal.reduce(out_2, wide, kind=MAX, dtype=u32)", "T.metal.reduce(out_3, partial)",
        ]


@case
class QuadReduce(MetalCase):
    path: Path = ROOT / "scheduled" / "quad_reduce.py"

    def inputs(self):
        return (utils.torch.arange(64, dtype=utils.torch.int32).reshape(16, 4),)

    def guard(self, state, problems):
        assert state.result.tolist() == [[16 * group + 6] for group in range(16)]
        assert issued(state.lowered) == ["T.metal.reduce(out, x, dtype=i32)"]


@case
class Prefix(MetalCase):
    path: Path = ROOT / "scheduled" / "prefix.py"

    def inputs(self):
        return LANE, WIDE

    def guard(self, state, problems):
        exclusive, inclusive = state.result
        assert exclusive.tolist() == [[t * (t - 1) // 2 for t in range(32)]] * 2
        assert inclusive[:, 7].tolist() == [(0xFFFFFFF0 + 21) % 2**32] * 2
        assert issued(state.lowered) == [
            "T.metal.prefix(out, x, dtype=i32)", "T.metal.prefix(out_1, wide, inclusive=True, dtype=u32)",
        ]


@case
class Shuffle(MetalCase):
    path: Path = ROOT / "scheduled" / "shuffle.py"

    def inputs(self):
        return LANE, WIDE, u16(1), u16(5)

    def guard(self, state, problems):
        down, broadcast, up = state.result
        assert down.tolist() == [[(t + 1) % 32 for t in range(32)]] * 2
        assert broadcast.tolist() == [[5] * 32] * 2
        assert up[:, 8].tolist() == [0xFFFFFFF0] * 2
        assert issued(state.lowered) == [
            "T.metal.shuffle(out, x, one, kind=ROTATE_DOWN, dtype=i32)",
            "T.metal.shuffle(out_1, x, five, kind=BROADCAST, dtype=i32)",
            "T.metal.shuffle(out_2, wide, one, kind=ROTATE_UP, dtype=u32)",
        ]


@case
class ShuffleFill(MetalCase):
    path: Path = ROOT / "scheduled" / "shuffle_fill.py"

    def inputs(self):
        return LANE, LANE + 100, u16(1)

    def guard(self, state, problems):
        assert state.result.tolist() == [[t + 107 if t % 8 == 0 else t - 1 for t in range(32)]] * 2
        assert issued(state.lowered) == ["T.metal.shuffle_fill(out, x, fill, delta, modulo=8, dtype=i32)"]


@case
class Ballot(MetalCase):
    path: Path = ROOT / "scheduled" / "ballot.py"

    def inputs(self):
        return (LANE % 2 == 0,)

    def guard(self, state, problems):
        assert state.result.tolist() == [[0x55555555]] * 2
        assert issued(state.lowered) == ["T.metal.ballot(out, even)"]


@case
class ShuffleOutside(MetalCase):
    path: Path = ROOT / "scheduled" / "shuffle_outside.py"
    evaluate_refusal: str = "outside the group"

    def inputs(self):
        return LANE, utils.torch.full((2, 32), 40, dtype=utils.torch.uint16)

    def guard(self, state, problems):
        assert issued(state.lowered) == ["T.metal.shuffle(out, x, lane, dtype=i32)"]


@case
class ShuffleFillOutside(MetalCase):
    path: Path = ROOT / "scheduled" / "shuffle_fill_outside.py"
    evaluate_refusal: str = "outside the group"

    def inputs(self):
        return LANE, LANE, u16(9)

    def guard(self, state, problems):
        assert issued(state.lowered) == ["T.metal.shuffle_fill(out, x, fill, delta, modulo=8, dtype=i32)"]


@case
class ShuffleFillModulo(MetalCase):
    path: Path = ROOT / "scheduled" / "shuffle_fill_modulo.py"
    refusal: str = "modulo <= width"


@case
class ReduceMemory(MetalCase):
    path: Path = ROOT / "scheduled" / "reduce_memory.py"
    refusal: str = "src: gmem does not match rmem"


@case
class Unmeshed(MetalCase):
    """Core evaluates the unscheduled product; lowering finds no mesh to issue it in."""

    path: Path = ROOT / "scheduled" / "unmeshed.py"
    finalize_refusal: str = "MatMul instruction issue has no lexical mesh"

    def inputs(self):
        return MATRICES[:2]

    def guard(self, state, problems):
        a, b = MATRICES[:2]
        assert utils.torch.equal(state.result, a @ b)


@case
class CoreReduce(MetalCase):
    path: Path = ROOT / "scheduled" / "core_reduce.py"
    finalize_refusal: str = "has no lowering for Reduce"


CASES = cases(ROOT / "hir", ROOT / "scheduled", case_type=MetalCase)
STAGES = {"hir": _hir, "scheduled": _scheduled}


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_schedule_case(case: MetalCase) -> None:
    problems = Problems(case)
    STAGES[case.path.parent.name](case, problems)
    problems.dump()
