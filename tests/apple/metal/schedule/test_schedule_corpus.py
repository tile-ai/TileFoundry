"""CUDA-style authored Case/guard checks of Metal candidate admission."""
from dataclasses import dataclass
from pathlib import Path
import pytest
from tests.core._corpus import Case, Problems, case, cases
from tilefoundry.analysis.check import check_program
from tilefoundry.ir.core import Call, get_registered_name
from tilefoundry.ir.types.layout import size
from tilefoundry.ir.visitor import collect_exprs, StmtVisitor
from tilefoundry.visitor_registry.verify import verify_module
from tilefoundry.schedule import candidates, finalize
from tilefoundry.analysis import analyze
from tilefoundry.evaluator import evaluate
from tilefoundry.utils.lazy import torch
from tilefoundry.visitor_registry.contexts import TypeInferContext
from tilefoundry.visitor_registry.valuation import ValuationVisitor

ROOT = Path(__file__).parents[1] / "fixtures"

@dataclass(frozen=True)
class MetalCase(Case):
    scheduled: bool = False
    explicit: bool = False
    threads: int = 64
    groups: int = 2

    def guard(self, state, problems):
        module, entry, report = state
        assert [(t.name, t.size) for t in module.topologies] == [("threadgroup", self.groups), ("thread", self.threads)]
        contexts = ValuationVisitor.index_contexts(entry, TypeInferContext(
            module=module, function=entry, target=module.resolve_target(), owner=entry))
        for call in collect_exprs(entry.body):
            if isinstance(call, Call):
                mesh = contexts[id(call)].current_mesh
                assert size(mesh.layout) == self.groups * self.threads > 1
        return report


def require(report, label, name):
    row, = [r for r in report["lines"] if r["line"].split(":")[0] == label]
    found = [c for c in row["candidates"] if c["id"] == "T.metal." + name]
    assert found, row
    return found


def refuse(report, label, name):
    row, = [r for r in report["lines"] if r["line"].split(":")[0] == label]
    assert all(c["id"] != "T.metal." + name for c in row["candidates"]), row
    assert any(c["id"] == "T.metal." + name and c["reason"] for c in row["refused"]), row

@case
class Matrix(MetalCase):
    path: Path = ROOT / "hir_matmul.py"
    groups: int = 4

    def guard(self, state, problems):
        report = super().guard(state, problems)
        for label in ("ordinary_mm", "odd_mm"):
            refuse(report, label, "matmul")

@case
class Collectives(MetalCase):
    path: Path = ROOT / "hir_collectives.py"
    threads: int = 128

    def guard(self, state, problems):
        report = super().guard(state, problems)
        for label, kind in (("total", "SUM"), ("maximum", "MAX"), ("minimum", "MIN")):
            matches = require(report, label, "reduce")
            assert any(kind in str(m["path"]) for m in matches), matches
        for label in ("mean", "ordinary_sum"):
            refuse(report, label, "reduce")

@case
class Memory(MetalCase):
    path: Path = ROOT / "hir_memory.py"

    def guard(self, state, problems):
        report = super().guard(state, problems)
        require(report, "load", "ldmatrix")
        require(report, "load_shared", "ldmatrix")
        require(report, "store", "stmatrix")
        refuse(report, "bad_load", "ldmatrix")

@case
class Explicit(MetalCase):
    threads: int = 128
    path: Path = ROOT / "metal_explicit.py"
    explicit: bool = True

    def guard(self, state, problems):
        module, entry, report = state
        assert report is None
        a = torch.eye(8).repeat(4,1,1) * 2
        b = torch.arange(256, dtype=torch.float32).reshape(4,8,8)
        c = torch.full_like(b, 3)
        actual = evaluate(entry, torch.zeros_like(b), a, b, c)
        assert torch.equal(actual, 2*b+c)
        assert torch.equal(c, torch.full_like(c, 3))
        assert [(t.name, t.size) for t in module.topologies] == [("threadgroup",2), ("thread",128)]
        lowered=finalize(analyze(module,entry,analysis=("memory",)))
        verify_module(lowered)
        names=[]
        class Ops(StmtVisitor):
            def visit_Evaluate(self,stmt,ctx=None):
                names.append(get_registered_name(type(stmt.callable)))
        Ops().visit(lowered.entry_function().body)
        assert names==["T.metal.matmul","T.metal.mma_m8n8k8"]

@case
class Scheduled(MetalCase):
    path: Path = ROOT / "metal_scheduled.py"
    scheduled: bool = True

    def guard(self, state, problems):
        super().guard(state, problems)
        module, entry, _ = state
        a = torch.eye(8).repeat(2,1,1)
        b = torch.arange(128, dtype=torch.float32).reshape(2,8,8)
        actual = evaluate(entry, torch.zeros_like(b), a, b)
        assert torch.equal(actual, 2*b)
        analyzed = analyze(module, entry, analysis=("memory",))
        lowered = finalize(analyzed)
        verify_module(lowered)
        names = []
        class Ops(StmtVisitor):
            def visit_Evaluate(self, stmt, ctx=None):
                names.append(get_registered_name(type(stmt.callable)))
        Ops().visit(lowered.entry_function().body)
        assert "T.metal.matmul" in names and "T.metal.mma_m8n8k8" in names

@case
class Quad(MetalCase):
    path: Path = ROOT / "hir_quad.py"
    threads: int = 32

    def guard(self,state,problems):
        report=super().guard(state,problems)
        matches=require(report,"total","reduce")
        assert any(any(p["name"]=="width" and p["value"]=="4" for p in m["path"]) for m in matches)

CASES = cases(ROOT, case_type=MetalCase)

@pytest.mark.parametrize("case", CASES, ids=lambda c: c.id)
def test_schedule_case(case):
    problems = Problems(case)
    module = problems.run("load", case.load)
    def check(m):
        return check_program(m, m.entry_function())
    checked = problems.run("check", check, module)
    report = None if case.explicit or case.scheduled else problems.run("candidates", lambda pair: candidates(*pair), checked)
    if checked is not None and (case.explicit or case.scheduled or report is not None):
        problems.run("guard", case.guard, (*checked, report), problems)
    problems.dump()
