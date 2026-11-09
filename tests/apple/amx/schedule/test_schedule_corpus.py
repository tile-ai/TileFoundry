"""Authored AMX HIR checked through the public candidate report, without execution."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from tests.core._corpus import Case, Problems, case, cases
from tilefoundry import analysis
from tilefoundry.interfaces import schedule, typeinfer, valuation
from tilefoundry.ir import core, expr, hir, types
from tilefoundry.schedule import candidates
from tilefoundry_apple.amx.ir import tir

ROOT = Path(__file__).parents[1] / "fixtures"


@dataclass(frozen=True)
class CandidateState:
    module: object
    entry: object
    candidates: dict | None


@dataclass(frozen=True)
class AmxCase(Case):
    explicit: bool = False
    workers: int = 12

    def guard(self, state, problems):
        assert [(t.name, t.size) for t in state.module.topologies] == [("thread", 12)]
        contexts = valuation.index_contexts(
            state.entry,
            typeinfer.TypeInferContext(module=state.module, function=state.entry,
                             target=state.module.resolve_target(), owner=state.entry),
        )
        calls = [node for node in core.collect_exprs(state.entry.body) if isinstance(node, expr.Call)]
        assert calls
        for call in calls:
            mesh = contexts[id(call)].current_mesh
            assert tuple(t.name for t in mesh.topologies) == ("thread",)
            assert types.size(mesh.layout) == self.workers > 1

        supported = {
            "ldx": "ldx_v2", "ldy": "ldy_v2", "extrx": "extrx_v2",
            "extry": "extry_v2", "matfp": "matfp_v2", "vecfp": "vecfp_v2",
            "matint": "matint_v1", "vecint": "vecint_v2", "genlut": "genlut_v2",
        }
        if state.candidates is None:
            names = {
                expr.get_registered_name(type(node.target.op))
                for node in core.collect_exprs(state.entry.body)
                if isinstance(node, expr.Call) and isinstance(node.target, hir.schedule.ScheduleOp)
            }
        else:
            names = {
                item["id"] for row in state.candidates["lines"]
                for group in ("candidates", "refused") for item in row[group]
            }
        for name in names:
            instruction = name.removeprefix("T.amx.")
            family = instruction.split("_v", 1)[0]
            if family in supported:
                assert instruction == supported[family], (name, supported[family])


def _row(state, label):
    rows = [row for row in state.candidates["lines"] if row["line"].split(":")[0] == label]
    assert len(rows) == 1, (label, state.candidates)
    return rows[0]


def _requires(state, label, instruction, **bindings):
    row = _row(state, label)
    fits = [fit for fit in row["candidates"] if fit["id"] == instruction]
    assert fits, (label, instruction, row)
    assert any(
        all(str(value) in dict((p["name"], p["value"]) for p in fit["path"]).get(key, "").split("|")
            for key, value in bindings.items())
        for fit in fits
    ), (label, instruction, bindings, fits)


def _rejects(state, label, instruction):
    row = _row(state, label)
    assert all(fit["id"] != instruction for fit in row["candidates"]), row
    assert any(fit["id"] == instruction and fit["reason"] for fit in row["refused"]), row


@case
class MatMul(AmxCase):
    workers: int = 8
    path: Path = ROOT / "hir_matmul.py"

    def guard(self, state, problems):
        super().guard(state, problems)
        for label, instruction, dtype in (
            ("half", "fma16", "f16"), ("single", "fma32", "f32"),
            ("double", "fma64", "f64"), ("integer", "mac16", "i16"),
        ):
            _requires(state, label, "T.amx." + instruction, mode="MATRIX", acc_dtype=dtype)
        _requires(state, "integer", "T.amx.matint_v1", alu="ADD_PRODUCT")
        _requires(state, "single", "T.amx.matfp_v2", alu="ADD_PRODUCT", fp_mode="F32")
        _rejects(state, "odd_m", "T.amx.fma32")
        _rejects(state, "transposed_axes", "T.amx.fma32")
        _rejects(state, "ordinary_memory", "T.amx.fma32")
        contexts = valuation.index_contexts(
            state.entry,
            typeinfer.TypeInferContext(module=state.module, function=state.entry,
                             target=state.module.resolve_target(), owner=state.entry),
        )
        for label, instruction, expected in (
            ("half", tir.Fma16, (1, 2, 2)), ("single", tir.Fma32, (1, 4, 2)),
            ("double", tir.Fma64, (1, 8, 2)),
        ):
            call = next(node for node in core.collect_exprs(state.entry.body)
                        if isinstance(node, expr.Call) and (expr.value_label(node) or "").split(":")[0] == label)
            frame = contexts[id(call)]
            attrs, operands = schedule.candidate_operands(call, instruction)
            matches = hir.schedule.match_instruction(instruction, attrs, operands, frame.current_mesh, frame=frame)
            assert matches
            for match in matches:
                _, projection = schedule.check_call(match.op, operands, frame.current_mesh, frame=frame)
                assert projection.repeat == expected, (label, projection.repeat)



@case
class Elementwise(AmxCase):
    workers: int = 4
    path: Path = ROOT / "hir_elementwise.py"

    def guard(self, state, problems):
        super().guard(state, problems)
        for label, instruction in (("half_mul", "fma16"), ("single_mul", "fma32"),
                                   ("double_mul", "fma64"), ("integer_mul", "mac16")):
            _requires(state, label, "T.amx." + instruction, mode="VECTOR", skip_z=True)
        for label, instruction in (("half_neg", "fms16"), ("single_neg", "fms32"),
                                   ("double_neg", "fms64")):
            _requires(state, label, "T.amx." + instruction, mode="VECTOR", skip_y=True, skip_z=True)
        _requires(state, "single_mul", "T.amx.vecfp_v2", alu="PRODUCT", fp_mode="F32")
        _requires(state, "integer_mul", "T.amx.vecint_v2", alu="PRODUCT")
        for label, alu in (("single_add", "ADD_X"), ("single_min", "MIN"), ("single_max", "MAX")):
            _requires(state, label, "T.amx.vecfp_v2", alu=alu, fp_mode="F32")
        _requires(state, "single_add", "T.amx.fma32", mode="VECTOR", skip_y=True, skip_z=False)
        _requires(state, "single_sub", "T.amx.fms32", mode="VECTOR", skip_y=True, skip_z=False)
        _requires(state, "integer_add", "T.amx.mac16", mode="VECTOR", skip_y=True, skip_z=False)
        _requires(state, "integer_sub", "T.amx.vecint_v2", alu="SUB_SUM", mask=5, shift=0)
        _rejects(state, "unsigned_mul", "T.amx.fma16")


@case
class Memory(AmxCase):
    workers: int = 2
    path: Path = ROOT / "hir_memory.py"

    def guard(self, state, problems):
        super().guard(state, problems)
        for instruction in ("ldx_v2", "ldy_v2", "ldz"):
            _requires(state, "load", "T.amx." + instruction, dtype="f32", count="ONE")
        _requires(state, "load_interleaved", "T.amx.ldzi", dtype="f32")
        for instruction in ("stx", "sty", "stz"):
            _requires(state, "store", "T.amx." + instruction, dtype="f32", count="ONE")
        _requires(state, "store_interleaved", "T.amx.stzi", dtype="f32")
        _rejects(state, "bad_load", "T.amx.ldx_v2")
        _rejects(state, "bad_store", "T.amx.stx")
        _requires(state, "identity_cast", "T.amx.extrx_v2", src_dtype="f32", dst_dtype="f32")
        _rejects(state, "identity_cast", "T.amx.extry_v2")
        _requires(state, "float_extract", "T.amx.extrx_v2", mode="FLOAT", src_dtype="f32", dst_dtype="f16")
        _requires(state, "integer_extract", "T.amx.extrx_v2", mode="INTEGER", src_dtype="i32", dst_dtype="i16")
        _requires(state, "vertical_extract", "T.amx.extry_v2", mode="FLOAT", src_dtype="f32", dst_dtype="f16")


        for label in ("float_cast", "integer_cast"):
            for instruction in ("extrx_v2", "extry_v2"):
                _rejects(state, label, "T.amx." + instruction)


@case
class Explicit(AmxCase):
    path: Path = ROOT / "amx_explicit.py"
    explicit: bool = True

    def guard(self, state, problems):
        super().guard(state, problems)
        assert state.candidates is None
        instructions = {
            expr.get_registered_name(type(node2.target.op)) for node2 in core.collect_exprs(state.entry.body)
            if isinstance(node2, expr.Call) and isinstance(node2.target, hir.schedule.ScheduleOp)
        }
        assert {"T.amx.set", "T.amx.clr", "T.amx.genlut_v2", "T.amx.extrx_v2", "T.amx.extry_v2"} <= instructions
        assert not any(name.startswith("tf.") for name in instructions)
        for node in core.collect_exprs(state.entry.body):
            if not isinstance(node, expr.Call):
                continue
            assert isinstance(node.target, hir.schedule.ScheduleOp)
            name = expr.get_registered_name(type(node.target.op))
            if name == "T.amx.genlut_v2":
                assert node.target.op.mode.name == "LOOKUP32_2"
                assert node.args[0].type.dtype.name == "u8"
                assert node.args[0].type.shape == (4,)
            if name in ("T.amx.extrx_v2", "T.amx.extry_v2"):
                assert node.target.op.mode.name == "COPY"


CASES = cases(ROOT, case_type=AmxCase)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_schedule_case(case: AmxCase) -> None:
    problems = Problems(case)
    module = problems.run("load", case.load)
    checked = problems.run("check", lambda m: analysis.check_program(m, m.entry_function()), module)
    report = None if case.explicit else problems.run("candidates", lambda pair: candidates(*pair), checked)
    if checked is not None:
        state = CandidateState(*checked, report)
        problems.run("guard", case.guard, state, problems)
    problems.dump()
