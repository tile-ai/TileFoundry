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

import tilefoundry
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


def inside_average(x, *window, **options):
    """An average pool of *x*: each window sum times the fp16 reciprocal of its cells in the map."""
    sums = utils.F.avg_pool2d(x.float(), *window, **options, divisor_override=1)
    inside = utils.F.avg_pool2d(utils.torch.ones(1, 1, *x.shape[2:]), *window, **options, divisor_override=1)
    return tir.evaluation.f16(sums * tir.evaluation.f16(1 / inside)).to(utils.torch.float16)


@dataclass(frozen=True)
class AneCase(Case):
    """One ANE fixture; a field names the error its stage must raise.

    ``refusal`` is the load-time VerifyError, ``finalize_refusal`` the LoweringError and
    ``emit_refusal`` the error code generation raises. ``runs`` marks a scheduled fixture
    that goes through the whole flow: every host generates its code, and a machine with a
    neural engine also compiles it, runs it and requires the result to equal the evaluator's.
    """

    refusal: str | None = None
    finalize_refusal: str | None = None
    emit_refusal: str | None = None
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


def _unemitted(case, lowered):
    with pytest.raises(ValueError, match=case.emit_refusal):
        device.emit(lowered)


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
        problems.run("emit", device.emit, lowered)
        if device.available():
            compiled = problems.run("compile", tilefoundry.compile, lowered)
            ran = problems.run("run", device.run, compiled, lowered, case.inputs())
            problems.expect("device result", utils.torch.equal, ran, result)
    if case.emit_refusal is not None:
        problems.run("emit refusal", _unemitted, case, lowered)
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
class ConvVariants(AneCase):
    """The engine's SiLU and sigmoid tables stay within 2**-6 and 2**-11 of the functions."""

    path: Path = ROOT / "scheduled" / "conv_variants.py"
    runs: bool = True

    def inputs(self):
        return (
            small((1, 32, 16, 16), 10), small((32, 32, 1, 1), 11), small((32,), 12, -3, 3), small((32, 32, 1, 1), 13),
            small((32, 32, 1, 1), 14), small((32, 32, 3, 3), 15), small((32, 1, 3, 3), 16),
        )

    def guard(self, state, problems):
        x, rectified, b, silu, sigmoid, padded, depthwise = self.inputs()
        relu, smooth, squashed, same, grouped = state.result.split(32, dim=1)
        assert utils.torch.equal(relu, utils.F.conv2d(x, rectified, b).relu())
        assert (smooth.float() - utils.F.silu(utils.F.conv2d(x, silu).float())).abs().max() <= 2**-6
        assert (squashed.float() - utils.F.conv2d(x, sigmoid).float().sigmoid()).abs().max() <= 2**-11
        assert utils.torch.equal(same, utils.F.conv2d(x, padded, padding=1))
        assert utils.torch.equal(grouped, utils.F.conv2d(x, depthwise, padding=1, groups=32))
        assert issued(state.lowered) == [
            "T.ane.load(x, t0)", "T.ane.conv(t0, rectified, t1, b, nonlinear=RELU)", "T.ane.store(t1, t2)",
            "T.ane.conv(t0, silu, t3, nonlinear=SILU)", "T.ane.store(t3, t4)",
            "T.ane.conv(t0, sigmoid, t5, nonlinear=SIGMOID)", "T.ane.store(t5, t6)",
            "T.ane.conv(t0, padded, t7, padding=(1, 1))", "T.ane.bypass(t7, t8)", "T.ane.store(t8, t9)",
            "T.ane.conv(t0, depthwise, t10, padding=(1, 1), groups=32)", "T.ane.store(t10, t11)",
        ]


@case
class MixedChain(AneCase):
    """Each engine reads the other's results from L2; a planar sum reads one result of each."""

    path: Path = ROOT / "scheduled" / "mixed_chain.py"
    runs: bool = True

    def inputs(self):
        return small((1, 16, 16, 16), 10), small((32, 16, 1, 1), 11), small((16, 32, 3, 3), 12)

    def guard(self, state, problems):
        x, v, w = (value.float() for value in self.inputs())
        h = utils.F.conv2d((x + 1) * 2, v)
        expected = utils.torch.cat((utils.F.conv2d(h + h.abs(), w), utils.F.conv2d(h, w)), dim=1)
        assert utils.torch.equal(state.result, expected.to(utils.torch.float16))
        assert issued(state.lowered) == [
            "T.ane.load(x, t0)", "T.ane.bypass(t0, t1)", "T.ane.unary(t1, t2, scale=2.0, bias=1.0)",
            "T.ane.conv(t2, v, t3)", "T.ane.unary(t3, t4, abs=True)", "T.ane.binary(t3, t4, t5, kind=ADD)",
            "T.ane.conv(t5, w, t6)", "T.ane.store(t6, t7)", "T.ane.conv(t3, w, t8)", "T.ane.store(t8, t9)",
        ]


@case
class Ffn(AneCase):
    """The four passes stay within 0.5 of the block computed with an exact SiLU."""

    path: Path = ROOT / "scheduled" / "ffn.py"
    runs: bool = True

    def inputs(self):
        return (small((1, 32, 16, 16), 10), small((32, 32, 1, 1), 11),
                small((32, 32, 1, 1), 12), small((32, 32, 1, 1), 13))

    def guard(self, state, problems):
        x, gate, up, down = (value.float() for value in self.inputs())
        hidden = utils.F.silu(utils.F.conv2d(x, gate)) * utils.F.conv2d(x, up)
        assert (state.result.float() - utils.F.conv2d(hidden, down)).abs().max() <= 0.5


@case
class ConvRelu(AneCase):
    path: Path = ROOT / "scheduled" / "conv_relu.py"

    def inputs(self):
        return small((1, 32, 16, 16), 0), small((32, 32, 3, 3), 1), small((32,), 2, -3, 3)

    def guard(self, state, problems):
        x, w, b = (value.float() for value in self.inputs())
        assert utils.torch.equal(state.result, utils.F.conv2d(x, w, b, padding=1).relu().to(utils.torch.float16))
        assert issued(state.lowered) == [
            "T.ane.load(x, t0)", "T.ane.conv(t0, w, t1, b, padding=(1, 1), nonlinear=RELU)", "T.ane.store(t1, out)"
        ]


@case
class MatMul(AneCase):
    """Row m of the [256, 16] matrix is row m % 8 of channel m // 8; w holds the right operand."""

    path: Path = ROOT / "scheduled" / "matmul.py"
    runs: bool = True

    def inputs(self):
        return small((256, 16), 0), small((32, 16, 1, 1), 1)

    def guard(self, state, problems):
        a, w = (value.float() for value in self.inputs())
        assert utils.torch.equal(state.result, (a @ w.flatten(1).T).to(utils.torch.float16))
        assert issued(state.lowered) == [
            "T.ane.load(a, t0)", "T.ane.bypass(t0, t1, perm=(2, 1, 0))", "T.ane.conv(t1, w, t2)",
            "T.ane.bypass(t2, t3, perm=(2, 1, 0))", "T.ane.store(t3, out)",
        ]


@case
class MatMulTiled(MatMul):
    """Four tiles of 16 channels share one weight constant; each stores into its output window."""

    path: Path = ROOT / "scheduled" / "matmul_tiled.py"

    def inputs(self):
        return small((1, 64, 8, 16), 0), small((16, 16, 1, 1), 1)

    def guard(self, state, problems):
        a, w = (value.float() for value in self.inputs())
        assert utils.torch.equal(state.result, (a @ w.flatten(1).T).to(utils.torch.float16))
        assert issued(state.lowered)[:5] == [
            "T.ane.load(t0, t1)", "T.ane.bypass(t1, t2, perm=(2, 1, 0))", "T.ane.conv(t2, w, t3)",
            "T.ane.bypass(t3, t4, perm=(2, 1, 0))", "T.ane.store(t4, t5)",
        ]


@case
class Chain(AneCase):
    """The first product stays in the on-chip buffer: the second reads it there."""

    path: Path = ROOT / "scheduled" / "chain.py"
    runs: bool = True

    def inputs(self):
        return small((1, 16, 8, 16), 0), small((16, 16, 1, 1), 1), small((16, 16, 1, 1), 2)

    def guard(self, state, problems):
        a, b, w = (value.float() for value in self.inputs())
        assert utils.torch.equal(state.result, (a @ b.flatten(1).T @ w.flatten(1).T).to(utils.torch.float16))
        assert issued(state.lowered) == [
            "T.ane.load(a, t0)", "T.ane.bypass(t0, t1, perm=(2, 1, 0))", "T.ane.conv(t1, b, t2)",
            "T.ane.conv(t2, w, t3)", "T.ane.bypass(t3, t4, perm=(2, 1, 0))", "T.ane.store(t4, out)",
        ]


@case
class TransposeHwc(AneCase):
    path: Path = ROOT / "scheduled" / "transpose_hwc.py"
    runs: bool = True

    def inputs(self):
        return ((utils.torch.arange(4096) - 2048).reshape(1, 32, 8, 16).to(utils.torch.float16),)

    def guard(self, state, problems):
        (x,) = self.inputs()
        assert utils.torch.equal(state.result, x.permute(0, 2, 3, 1))
        assert issued(state.lowered) == [
            "T.ane.load(x, t0)", "T.ane.bypass(t0, t1, perm=(1, 2, 0))", "T.ane.store(t1, out)"
        ]


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
            "T.ane.load(x, t0)",
            "T.ane.pool(t0, t1, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), kind=AVG)",
            "T.ane.store(t1, out)",
        ]


@case
class PoolVariants(AneCase):
    """Padding and an overhang take no part in a max: every input is negative."""

    path: Path = ROOT / "scheduled" / "pool_variants.py"
    runs: bool = True

    def inputs(self):
        return (small((1, 32, 16, 16), 0, -7, -1),)

    def guard(self, state, problems):
        (x,) = self.inputs()
        m = utils.F.max_pool2d(x, 2, 2)
        expected = utils.torch.cat((
            (m + 1) * 2, inside_average(m, 3, 1, 1), utils.F.max_pool2d(x, 3, 2, 1),
            utils.F.max_pool2d(x, 3, 2, ceil_mode=True), inside_average(x, 3, 2, ceil_mode=True),
        ), dim=1)
        assert utils.torch.equal(state.result, expected)
        assert issued(state.lowered) == [
            "T.ane.load(x, t0)", "T.ane.pool(t0, t1, kernel_size=(2, 2), stride=(2, 2), kind=MAX)",
            "T.ane.unary(t1, t2, scale=2.0, bias=1.0)", "T.ane.store(t2, t3)",
            "T.ane.pool(t1, t4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), count_include_pad=False, kind=AVG)",
            "T.ane.store(t4, t5)",
            "T.ane.pool(t0, t6, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), kind=MAX)", "T.ane.store(t6, t7)",
            "T.ane.pool(t0, t8, kernel_size=(3, 3), stride=(2, 2), ceil_mode=True, kind=MAX)", "T.ane.store(t8, t9)",
            "T.ane.pool(t0, t10, kernel_size=(3, 3), stride=(2, 2), ceil_mode=True, count_include_pad=False, kind=AVG)",
            "T.ane.store(t10, t11)",
        ]


@case
class Unary(AneCase):
    path: Path = ROOT / "scheduled" / "unary.py"
    runs: bool = True

    def inputs(self):
        return (small((1, 32, 16, 16), 0, -3, 3),)

    def guard(self, state, problems):
        (x,) = self.inputs()
        clipped = (2 * x + 3).clamp(0, 6)
        assert utils.torch.equal(state.result, (((clipped - 2).abs().clamp(0, 3) + 1) * 2 - 5) ** 2)
        assert issued(state.lowered) == [
            "T.ane.load(x, t0)", "T.ane.unary(t0, t1, pre_scale=2.0, bias=3.0, clamp=(0.0, 6.0))",
            "T.ane.unary(t1, t2, bias=-2.0, abs=True, clamp=(0.0, 3.0))", "T.ane.unary(t2, t3, scale=2.0, bias=1.0)",
            "T.ane.unary(t3, t4, bias=-5.0, square=True)", "T.ane.bypass(t4, t5)", "T.ane.store(t5, out)",
        ]


@case
class UnaryTiled(AneCase):
    """Each tile is a load, a planar pass and a store into its window of the output."""

    path: Path = ROOT / "scheduled" / "unary_tiled.py"
    runs: bool = True

    def inputs(self):
        return (small((1, 64, 128, 128), 5, -7, 7),)

    def guard(self, state, problems):
        (x,) = self.inputs()
        assert utils.torch.equal(state.result, x + 1)
        assert issued(state.lowered)[:3] == [
            "T.ane.load(t0, t1)", "T.ane.unary(t1, t2, bias=1.0)", "T.ane.store(t2, t3)"
        ]


@case
class Binary(AneCase):
    path: Path = ROOT / "scheduled" / "binary.py"
    runs: bool = True

    def inputs(self):
        return small((1, 32, 16, 16), 0, -3, 3), small((1, 32, 16, 16), 1, -3, 3)

    def guard(self, state, problems):
        x, y = self.inputs()
        h = (x + y).abs()
        assert utils.torch.equal(state.result, utils.torch.cat((y - x, h + (h - 2), (h * (h - 2) + 1) * 2), dim=1))
        assert issued(state.lowered) == [
            "T.ane.load(y, t0)", "T.ane.load(x, t1)", "T.ane.binary(t0, t1, t2, kind=SUB)", "T.ane.store(t2, t3)",
            "T.ane.binary(t1, t0, t4, kind=ADD)", "T.ane.unary(t4, t5, abs=True)", "T.ane.unary(t5, t6, bias=-2.0)",
            "T.ane.binary(t5, t6, t7, kind=ADD)", "T.ane.store(t7, t8)",
            "T.ane.binary(t5, t6, t9, kind=MUL)", "T.ane.unary(t9, t10, scale=2.0, bias=1.0)", "T.ane.store(t10, t11)",
        ]


@case
class ReduceVariants(AneCase):
    """The mean is the sum scaled by 1/256; the engine squares each element before it adds."""

    path: Path = ROOT / "scheduled" / "reduce_variants.py"
    runs: bool = True

    def inputs(self):
        return (small((1, 32, 16, 16), 0, -2, 2),)

    def guard(self, state, problems):
        (x,) = self.inputs()
        total = x.float().sum((2, 3), keepdim=True)
        mean = tir.evaluation.f16(total * tir.evaluation.f16(utils.torch.tensor(1 / 256)))
        squares = (x.float() ** 2).sum((2, 3), keepdim=True)
        expected = utils.torch.cat((total, (mean + 1) * 2, squares), dim=1)
        assert utils.torch.equal(state.result, expected.to(utils.torch.float16))
        assert issued(state.lowered) == [
            "T.ane.load(x, t0)", "T.ane.reduce(t0, t1, axes=(2, 3), keepdim=True, kind=SUM)", "T.ane.store(t1, t2)",
            "T.ane.reduce(t0, t3, axes=(2, 3), keepdim=True, kind=SUM, scale=0.00390625)",
            "T.ane.unary(t3, t4, scale=2.0, bias=1.0)", "T.ane.store(t4, t5)",
            "T.ane.reduce(t0, t6, axes=(2, 3), keepdim=True, kind=SUM, square=True)", "T.ane.store(t6, t7)",
        ]


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
        """The value the entry must return, or None for a case that is refused."""
        return None

    def guard(self, state, problems):
        if self.expected() is not None:
            assert utils.torch.equal(state.result, self.expected().to(utils.torch.float16))


VALUES = ROOT / "ops" / "values.py"
REFUSED = ROOT / "ops" / "refused.py"
UNPLANNED = ROOT / "ops" / "unplanned.py"


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
        return inside_average(x, 3, 2, 1)


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
class Unloaded(OpCase):
    path: Path = REFUSED
    name: str | None = "Unloaded"
    refusal: str = "src: gmem does not match smem"


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


@case
class StoreLoaded(OpCase):
    path: Path = UNPLANNED
    name: str | None = "StoreLoaded"
    emit_refusal: str = "T.ane.store: its source must be the result of a computation"


@case
class StoreRead(OpCase):
    path: Path = UNPLANNED
    name: str | None = "StoreRead"
    emit_refusal: str = "T.ane.store: its source is read again after being written out"


@case
class Unstored(OpCase):
    path: Path = UNPLANNED
    name: str | None = "Unstored"
    emit_refusal: str = "the output must be written by T.ane.store"


@case
class StoredTwice(OpCase):
    path: Path = UNPLANNED
    name: str | None = "StoredTwice"
    emit_refusal: str = "an L2 tensor is written out once"


@case
class Copied(OpCase):
    path: Path = UNPLANNED
    name: str | None = "Copied"
    finalize_refusal: str = "needs a copy; store the update into the window instead"


CASES = (*cases(ROOT / "hir", ROOT / "scheduled", case_type=AneCase), *special(OpCase))
STAGES = {"hir": _hir, "scheduled": _scheduled, "ops": _scheduled}


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_schedule_case(case: AneCase) -> None:
    problems = Problems(case)
    STAGES[case.path.parent.name](case, problems)
    problems.dump()
