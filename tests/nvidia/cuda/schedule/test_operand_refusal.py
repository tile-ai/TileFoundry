"""Scheduling operand refusal checks."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from tests.core._corpus import load_program
from tests.nvidia.cuda.schedule._support import (
    _OperandMatchVisitor,
    entry_function,
)
from tilefoundry.inspection import PatternPrinter
from tilefoundry.ir.core import (
    Var,
)
from tilefoundry.ir.pattern import (
    PatternMatcher,
)
from tilefoundry.ir.tir.stmts import Evaluate
from tilefoundry.ir.types import (
    StorageKind,
)
from tilefoundry.ir.visitor import StmtVisitor
from tilefoundry.visitor_registry.instruction import ScheduleRefusal
from tilefoundry_nvidia.cuda.ir.tir.nn.wgmma import Wgmma


def test_operand_match_refusal_names_the_failed_pattern() -> None:
    function = entry_function(
        load_program(
            Path(__file__).parents[1] / "fixtures" / "schedule" / "tir" / "wgmma_a_k_major.py"
        )
    )
    rejected = None

    class RejectOne(StmtVisitor[None]):
        def visit_Evaluate(self, stmt: Evaluate) -> None:
            nonlocal rejected
            if rejected is not None or not isinstance(getattr(stmt.callable, "atom", None), Wgmma):
                return
            op = stmt.callable
            lhs = next(param for param in type(op)._op_schema.signature if param.name == "lhs")
            pattern = lhs.pattern.read_on(op)
            matcher = PatternMatcher(dict(op.atom.bindings))
            assert not matcher.match(pattern, replace(stmt.args[1].type, storage=StorageKind.GMEM))
            rejected = PatternPrinter().refusal(matcher.refusal)

    RejectOne().visit(function.body)
    assert rejected is not None
    assert "gmem" in rejected
    assert "smem" in rejected


@pytest.mark.parametrize(
    "row_scale, leading, position", ((2, None, "a_stride_00"), (1, 1, "a_stride_11"))
)
def test_wgmma_stride_refusal_names_operand_and_descriptor_position(
    row_scale, leading, position
) -> None:
    function = entry_function(
        load_program(
            Path(__file__).parents[1] / "fixtures" / "schedule" / "tir" / "wgmma_a_k_major.py"
        )
    )

    class RejectOne(_OperandMatchVisitor):
        rejected = False

        def visit_Evaluate(self, stmt: Evaluate) -> None:
            if self.rejected or not isinstance(getattr(stmt.callable, "atom", None), Wgmma):
                return
            lhs = stmt.args[1].type
            shard = lhs.layout
            outer = shard.layout
            row, column = outer.strides
            column = (column[0] if leading is None else leading, column[1])
            bad = replace(outer, strides=((row[0], row[1] * row_scale), column))
            operand = Var("bad_lhs", type=replace(lhs, layout=replace(shard, layout=bad)))
            with pytest.raises(ScheduleRefusal, match=rf"lhs:.*{position}"):
                super().visit_Evaluate(replace(stmt, args=(stmt.args[0], operand, stmt.args[2])))
            self.rejected = True

    visitor = RejectOne()
    visitor.visit(function.body)
    assert visitor.rejected
