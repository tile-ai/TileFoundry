"""ANE HIR offered to the candidate report; hand-scheduled ANE IR evaluated and lowered.

A fixture under ``hir/`` is checked through the candidate report. A fixture under
``scheduled/``, and each module of a fixture under ``ops/``, is loaded, type-checked,
evaluated by core's HIR evaluator, lowered, printed as TIR and reloaded; lowered TIR
is not evaluated.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from tests.apple.ane import device
from tests.core._corpus import Case, Problems, case, cases, reload, special
from tilefoundry import analysis, inspection, utils
from tilefoundry.interfaces import evaluator, schedule
from tilefoundry.ir import core, expr
from tilefoundry.passes import transforms
from tilefoundry.schedule import candidates, finalize
from tilefoundry_apple.ane.ir import tir

ROOT = Path(__file__).parents[1] / "fixtures"


def small(shape, seed, low=-1, high=1):
    """An f16 tensor of *shape* whose entries are integers in [low, high]."""
    generator = utils.torch.Generator().manual_seed(seed)
    return utils.torch.randint(low, high + 1, shape, generator=generator).to(utils.torch.float16)


@dataclass(frozen=True)
class AneCase(Case):
    """One ANE fixture; a field names the error its stage must raise.

    ``refusal`` is the load-time VerifyError and ``finalize_refusal`` the LoweringError.
    ``runs`` marks a scheduled fixture that goes through compile and run: it is compiled
    into a module on every host and, when this machine has a neural engine, run and
    required to equal the evaluator's result.
    """

    refusal: str | None = None
    finalize_refusal: str | None = None
    runs: bool = False

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


def _rows(state, label):
    return [row for row in state.candidates["lines"] if row["line"].split(":")[0] == label]


def _row(state, label):
    rows = _rows(state, label)
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


def _attributes(state, label, instruction):
    """The attributes the HIR match gives *instruction* for the call labelled *label*."""
    (call,) = (
        node for node in core.collect_exprs(state.entry.body)
        if isinstance(node, expr.Call) and (expr.value_label(node) or "").split(":")[0] == label
    )
    attributes, _ = schedule.candidate_operands(call, instruction)
    return attributes


def _no_ane(state, label):
    """No ``T.ane.*`` instruction lists *label*, whether its row refuses them all or is absent.

    A call whose HIR op has no instruction at all has no row.
    """
    for row in _rows(state, label):
        assert all(not fit["id"].startswith("T.ane.") for fit in row["candidates"]), row


class _Issued(core.StmtVisitor[None]):
    def visit_Evaluate(self, stmt, found):
        found.append(stmt)


def _statements(lowered: core.Module) -> list[core.Evaluate]:
    statements: list[core.Evaluate] = []
    _Issued().visit(lowered.entry_function(), statements)
    return statements


def _attribute(op, declared):
    value = getattr(op, declared.name)
    if declared.has_default and value == declared.default:
        return None
    return f"{declared.name}={getattr(value, 'name', value)}"


def issued(lowered: core.Module) -> list[str]:
    """Each statement the lowered entry issues: its Op, operands and non-default attributes.

    An operand is a parameter's name or ``t<i>``, a temporary numbered by its first use.
    """
    params = {param.name for param in lowered.entry_function().params}
    temporaries: dict[str, str] = {}

    def operand(arg):
        return arg.name if arg.name in params else temporaries.setdefault(arg.name, f"t{len(temporaries)}")

    described = []
    for stmt in _statements(lowered):
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
    return evaluator.evaluate(module, *(value.clone() for value in case.inputs()))


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
        reprinted = problems.run("reprint TIR", inspection.as_script, lowered)
        problems.expect("reprinted TIR", lambda text: text == printed, reprinted)
    if case.runs:
        compiled = problems.run("compile", device.compile, lowered)
        if compiled is not None and device.available():
            ran = problems.run("run", device.run, compiled, lowered, case.inputs())
            problems.expect("device result", utils.torch.equal, ran, result)
    problems.run("guard", case.guard, ScheduledState(module, result, lowered), problems)


@case
class ConvCandidates(AneCase):
    path: Path = ROOT / "hir" / "conv.py"

    def guard(self, state, problems):
        _requires(state, "same", "T.ane.conv", stride=(1, 1))
        _requires(state, "strided", "T.ane.conv", stride=(2, 2))
        _requires(state, "grouped", "T.ane.conv", stride=(1, 1), groups=2)
        _rejects(state, "single", "T.ane.conv", "no applicable HIR declaration")
        _rejects(state, "dilated", "T.ane.conv", "dilation")
        _rejects(state, "wide_step", "T.ane.conv", "stride")
        _rejects(state, "wide_pad", "T.ane.conv", "padding")


@case
class MatMulCandidates(AneCase):
    path: Path = ROOT / "hir" / "matmul.py"

    def guard(self, state, problems):
        _requires(state, "product", "T.ane.conv")
        _rejects(state, "rows", "T.ane.conv", "layout")
        _rejects(state, "single", "T.ane.conv", "no applicable HIR declaration")


@case
class ActivationCandidates(AneCase):
    path: Path = ROOT / "hir" / "activation.py"

    def guard(self, state, problems):
        _requires(state, "rectified", "T.ane.bypass", nonlinear="RELU")
        _no_ane(state, "squashed")
        _rejects(state, "single", "T.ane.bypass", "no applicable HIR declaration")


@case
class PoolCandidates(AneCase):
    path: Path = ROOT / "hir" / "pool.py"

    def guard(self, state, problems):
        for label in ("maximum", "average", "padded", "ceiled", "ceiled_valid"):
            _requires(state, label, "T.ane.pool")
        _rejects(state, "wide_step", "T.ane.pool", "stride")
        _rejects(state, "ceiled_average", "T.ane.pool", "no applicable HIR declaration")


@case
class UnaryCandidates(AneCase):
    """The report path omits the attributes a match fixes, so they are read from the pairing."""

    path: Path = ROOT / "hir" / "unary.py"

    def guard(self, state, problems):
        for label, fixed in (("absolute", {"abs": True}), ("squared", {"square": True}), ("negated", {"scale": -1.0})):
            _requires(state, label, "T.ane.unary")
            assert _attributes(state, label, tir.Unary) == fixed, label
        _no_ane(state, "rounded")


@case
class BinaryCandidates(AneCase):
    path: Path = ROOT / "hir" / "binary.py"

    def guard(self, state, problems):
        for label, kind in (
            ("added", "ADD"), ("subtracted", "SUB"), ("multiplied", "MUL"), ("smaller", "MIN"), ("larger", "MAX"),
        ):
            _requires(state, label, "T.ane.binary", kind=kind)
        _rejects(state, "divided", "T.ane.binary", "kind")
        _rejects(state, "broadcast", "T.ane.binary", "shape")
        _rejects(state, "single", "T.ane.binary", "no applicable HIR declaration")


@case
class ReduceCandidates(AneCase):
    path: Path = ROOT / "hir" / "reduce.py"

    def guard(self, state, problems):
        for label in ("total", "largest", "smallest", "width", "flat"):
            _requires(state, label, "T.ane.reduce")
        _rejects(state, "mean", "T.ane.reduce", "kind")
        _rejects(state, "channels", "T.ane.reduce", "axes")
        _rejects(state, "height", "T.ane.reduce", "axes")


@case
class TransposeCandidates(AneCase):
    path: Path = ROOT / "hir" / "transpose.py"

    def guard(self, state, problems):
        _requires(state, "swapped", "T.ane.bypass", perm=(2, 1, 0))
        _requires(state, "rotated", "T.ane.bypass", perm=(1, 2, 0))
        _rejects(state, "batch", "T.ane.bypass", "perm")


@case
class Conv1x1(AneCase):
    path: Path = ROOT / "scheduled" / "conv_1x1.py"
    runs: bool = True

    def inputs(self):
        return small((1, 32, 16, 16), 10), small((32, 32, 1, 1), 11)

    def guard(self, state, problems):
        x, w = self.inputs()
        assert utils.torch.equal(state.result, utils.F.conv2d(x, w))


@case
class Conv1x1Relu(Conv1x1):
    path: Path = ROOT / "scheduled" / "conv_1x1_relu.py"

    def guard(self, state, problems):
        x, w = self.inputs()
        assert utils.torch.equal(state.result, utils.F.conv2d(x, w).relu())


@case
class ConvRelu(AneCase):
    path: Path = ROOT / "scheduled" / "conv_relu.py"

    def inputs(self):
        return small((1, 32, 16, 16), 0), small((32, 32, 3, 3), 1), small((32,), 2, -3, 3)

    def guard(self, state, problems):
        x, w, b = (value.float() for value in self.inputs())
        assert utils.torch.equal(state.result, utils.F.conv2d(x, w, b, padding=1).relu().to(utils.torch.float16))
        assert issued(state.lowered) == ["T.ane.conv(out, x, w, b, padding=(1, 1), nonlinear=RELU)"]


@case
class MatMul(AneCase):
    path: Path = ROOT / "scheduled" / "matmul.py"

    def inputs(self):
        return small((8, 16), 0), small((16, 16), 1)

    def guard(self, state, problems):
        a, b = (value.float() for value in self.inputs())
        assert utils.torch.equal(state.result, (a @ b).to(utils.torch.float16))
        assert issued(state.lowered) == ["T.ane.bypass(t0, a, perm=(2, 1, 0))", "T.ane.conv(out, t0, b)"]


@case
class Chain(AneCase):
    """The first product stays in the on-chip buffer: the second reads it there, no copy out."""

    path: Path = ROOT / "scheduled" / "chain.py"

    def inputs(self):
        return small((8, 16), 0), small((16, 16), 1), small((16, 16), 2)

    def guard(self, state, problems):
        a, b, w = (value.float() for value in self.inputs())
        assert utils.torch.equal(state.result, ((a @ b) @ w).to(utils.torch.float16))
        assert issued(state.lowered) == ["T.ane.conv(t0, a, b)", "T.ane.conv(out, t0, w)"]
        first, second = _statements(state.lowered)
        assert first.args[0].type.storage is expr.StorageKind.SMEM
        assert second.args[1].type.storage is expr.StorageKind.SMEM


@case
class Transpose(AneCase):
    path: Path = ROOT / "scheduled" / "transpose.py"
    runs: bool = True

    def inputs(self):
        return ((utils.torch.arange(4096) - 2048).reshape(1, 32, 8, 16).to(utils.torch.float16),)

    def guard(self, state, problems):
        (x,) = self.inputs()
        assert utils.torch.equal(state.result, x.permute(0, 3, 2, 1))
        assert issued(state.lowered) == ["T.ane.bypass(out, x, perm=(2, 1, 0))"]


@case
class Pool(AneCase):
    """An average multiplies the window sum by the fp16 reciprocal of the window size."""

    path: Path = ROOT / "scheduled" / "pool.py"

    def inputs(self):
        return (small((1, 32, 16, 16), 0, -3, 3),)

    def guard(self, state, problems):
        (x,) = self.inputs()
        sums = utils.F.avg_pool2d(x.float(), 3, 2, 1, divisor_override=1)
        reciprocal = tir.evaluation.f16(utils.torch.tensor(1 / 9))
        assert utils.torch.equal(state.result, tir.evaluation.f16(sums * reciprocal).to(utils.torch.float16))
        assert issued(state.lowered) == [
            "T.ane.pool(out, x, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), kind=AVG)"
        ]


@case
class PoolMax(AneCase):
    path: Path = ROOT / "scheduled" / "pool_max.py"
    runs: bool = True

    def inputs(self):
        return (small((1, 32, 16, 16), 0, -3, 3),)

    def guard(self, state, problems):
        (x,) = self.inputs()
        assert utils.torch.equal(state.result, utils.F.max_pool2d(x, 2, 2))
        assert issued(state.lowered) == ["T.ane.pool(out, x, kernel_size=(2, 2), stride=(2, 2), kind=MAX)"]


@case
class Unary(AneCase):
    path: Path = ROOT / "scheduled" / "unary.py"
    runs: bool = True

    def inputs(self):
        return (small((1, 32, 16, 16), 0, -3, 3),)

    def guard(self, state, problems):
        (x,) = self.inputs()
        assert utils.torch.equal(state.result, (2 * x.float() + 3).clamp(0, 6).to(utils.torch.float16))
        assert issued(state.lowered) == ["T.ane.unary(out, x, pre_scale=2.0, bias=3.0, clamp=(0.0, 6.0))"]


@case
class UnaryChain(AneCase):
    path: Path = ROOT / "scheduled" / "unary_chain.py"
    runs: bool = True

    def inputs(self):
        return (small((1, 32, 16, 16), 21, -7, 7),)

    def guard(self, state, problems):
        (x,) = self.inputs()
        assert utils.torch.equal(state.result, (x.abs() + 1) * 2)


@case
class Binary(AneCase):
    path: Path = ROOT / "scheduled" / "binary.py"
    runs: bool = True

    def inputs(self):
        return small((1, 32, 16, 16), 0, -3, 3), small((1, 32, 16, 16), 1, -3, 3)

    def guard(self, state, problems):
        x, y = self.inputs()
        assert utils.torch.equal(state.result, x + y)
        assert issued(state.lowered) == ["T.ane.binary(out, x, y, kind=ADD)"]


@case
class Reduce(AneCase):
    """A sum over the spatial axes scaled by the fp16 reciprocal of their extent is their mean."""

    path: Path = ROOT / "scheduled" / "reduce.py"

    def inputs(self):
        return (small((1, 32, 16, 16), 0),)

    def guard(self, state, problems):
        (x,) = self.inputs()
        total = tir.evaluation.f16(x.float().sum((2, 3), keepdim=True))
        reciprocal = tir.evaluation.f16(utils.torch.tensor(1 / 256))
        assert utils.torch.equal(state.result, tir.evaluation.f16(total * reciprocal).to(utils.torch.float16))
        assert issued(state.lowered) == [
            "T.ane.reduce(out, x, axes=(2, 3), keepdim=True, kind=SUM, scale=0.00390625)"
        ]


@case
class ReduceSum(AneCase):
    path: Path = ROOT / "scheduled" / "reduce_sum.py"
    runs: bool = True

    def inputs(self):
        return (small((1, 32, 16, 16), 0, -3, 3),)

    def guard(self, state, problems):
        (x,) = self.inputs()
        assert utils.torch.equal(state.result, x.sum((2, 3), keepdim=True))
        assert issued(state.lowered) == ["T.ane.reduce(out, x, axes=(2, 3), keepdim=True, kind=SUM)"]


@case
class Core(AneCase):
    path: Path = ROOT / "scheduled" / "core.py"
    finalize_refusal: str = "has no lowering for Copy"


def grid(shape, modulus, offset):
    """An f16 tensor of *shape* whose entries cycle through *modulus* integers from *offset*."""
    count = utils.torch.tensor(shape).prod().item()
    return (utils.torch.arange(count) % modulus + offset).reshape(shape).to(utils.torch.float16)


class OpCase(AneCase):
    """One module of a fixture under ``ops/``; ``expected`` is the torch value of its result."""

    def expected(self):
        """The value the entry must return, or None for a case that a load refuses."""
        return None

    def guard(self, state, problems):
        assert utils.torch.equal(state.result, self.expected().to(utils.torch.float16))


VALUES = ROOT / "ops" / "values.py"
REFUSED = ROOT / "ops" / "refused.py"


@case
class UnaryOrder(OpCase):
    """The last input makes ``x + 1`` an fp16 tie, rounded before the square."""

    path: Path = VALUES
    name: str | None = "UnaryOrder"

    def inputs(self):
        return (utils.torch.tensor([-4, -3, -2, -1, 0, 1, 2, 2**-11]).reshape(1, 1, 1, 8).to(utils.torch.float16),)

    def expected(self):
        return utils.torch.tensor([9, 4, 1, 0, 1, 4, 9, 1 + 2**-9]).reshape(1, 1, 1, 8)


@case
class MaxPoolCeil(OpCase):
    path: Path = VALUES
    name: str | None = "MaxPoolCeil"

    def inputs(self):
        return (grid((1, 2, 5, 6), 11, -5),)

    def expected(self):
        (x,) = self.inputs()
        return utils.F.max_pool2d(x.float(), 3, 2, ceil_mode=True)


@case
class AveragePoolInside(OpCase):
    """Each window sum is multiplied by the fp16 reciprocal of the cells it has inside the map."""

    path: Path = VALUES
    name: str | None = "AveragePoolInside"

    def inputs(self):
        return (grid((1, 2, 5, 6), 11, -5),)

    def expected(self):
        (x,) = self.inputs()
        sums = utils.F.avg_pool2d(x.float(), 3, 2, 1, divisor_override=1)
        inside = utils.F.avg_pool2d(utils.torch.ones(1, 1, 5, 6), 3, 2, 1, divisor_override=1)
        return tir.evaluation.f16(sums * tir.evaluation.f16(1 / inside))


@case
class ReduceDropped(OpCase):
    path: Path = VALUES
    name: str | None = "ReduceDropped"

    def inputs(self):
        return (grid((1, 2, 3, 4), 5, 1),)

    def expected(self):
        (x,) = self.inputs()
        return x.sum((2, 3))


@case
class ReduceNorm(OpCase):
    path: Path = VALUES
    name: str | None = "ReduceNorm"

    def inputs(self):
        return (grid((1, 2, 3, 4), 5, 1),)

    def expected(self):
        (x,) = self.inputs()
        return tir.evaluation.f16((x.float() ** 2).sum((1, 3), keepdim=True).rsqrt())


@case
class ConvStridedGrouped(OpCase):
    path: Path = VALUES
    name: str | None = "ConvStridedGrouped"

    def inputs(self):
        return grid((1, 4, 6, 6), 7, -3), grid((6, 2, 3, 3), 5, -2), grid((6,), 4, -1)

    def expected(self):
        x, w, b = (value.float() for value in self.inputs())
        return utils.F.conv2d(x, w, b, stride=2, padding=1, groups=2)


@case
class MatrixBias(OpCase):
    path: Path = VALUES
    name: str | None = "MatrixBias"

    def inputs(self):
        return grid((8, 16), 9, -4), grid((16, 5), 7, -3), grid((5,), 4, -1)

    def expected(self):
        a, w, b = (value.float() for value in self.inputs())
        return a @ w + b


@case
class Gapped(OpCase):
    path: Path = REFUSED
    name: str | None = "Gapped"
    refusal: str = "src: could not satisfy"


@case
class ClampReversed(OpCase):
    path: Path = REFUSED
    name: str | None = "ClampReversed"
    refusal: str = "clamp_low <= clamp_high"


@case
class KeepdimWidth(OpCase):
    path: Path = REFUSED
    name: str | None = "KeepdimWidth"
    refusal: str = "keepdim: could not satisfy"


@case
class WeightOnChip(OpCase):
    path: Path = REFUSED
    name: str | None = "WeightOnChip"
    refusal: str = "weight: smem does not match gmem"


@case
class WrongResultShape(OpCase):
    path: Path = REFUSED
    name: str | None = "WrongResultShape"
    refusal: str = "differs from the pooled shape"


@case
class GroupsUneven(OpCase):
    path: Path = REFUSED
    name: str | None = "GroupsUneven"
    refusal: str = "groups 3 must divide the input channels 4"


CASES = (*cases(ROOT / "hir", ROOT / "scheduled", case_type=AneCase), *special(OpCase))
STAGES = {"hir": _hir, "scheduled": _scheduled, "ops": _scheduled}


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_schedule_case(case: AneCase) -> None:
    problems = Problems(case)
    STAGES[case.path.parent.name](case, problems)
    problems.dump()
