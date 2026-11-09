"""Scheduling operand refusal checks."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from tests.nvidia.cuda.schedule._support import (
    _OperandMatchVisitor,
    entry_function,
)
from tilefoundry import cli, inspection
from tilefoundry.interfaces import schedule
from tilefoundry.ir import core, expr, pattern
from tilefoundry_nvidia.cuda.ir.tir import nn


def test_operand_refusal_names_failed_pattern() -> None:
    function = entry_function(
        cli.load_authored_ir(
            str(Path(__file__).parents[1] / "fixtures" / "schedule" / "tir" / "wgmma_a_k_major.py")
        )
    )
    rejected = None

    class RejectOne(core.StmtVisitor[None]):
        def visit_Evaluate(self, stmt: core.Evaluate) -> None:
            nonlocal rejected
            if rejected is not None or not isinstance(getattr(stmt.callable, "atom", None), nn.Wgmma):
                return
            op = stmt.callable
            lhs = next(param for param in expr.schema_of(type(op)).signature if param.name == "lhs")
            declared = lhs.pattern.read_on(op)
            matcher = pattern.Matcher(dict(op.atom.bindings))
            assert not matcher.match(declared, replace(stmt.args[1].type, storage=expr.StorageKind.GMEM))
            rejected = inspection.PatternPrinter().refusal(matcher.refusal)

    RejectOne().visit(function.body)
    assert rejected is not None
    assert "gmem" in rejected
    assert "smem" in rejected


@pytest.mark.parametrize(
    "row_scale, leading, position", ((2, None, "a_stride_00"), (1, 1, "a_stride_11"))
)
def test_wgmma_stride_refusal_names_position(
    row_scale, leading, position
) -> None:
    function = entry_function(
        cli.load_authored_ir(
            str(Path(__file__).parents[1] / "fixtures" / "schedule" / "tir" / "wgmma_a_k_major.py")
        )
    )

    class RejectOne(_OperandMatchVisitor):
        rejected = False

        def visit_Evaluate(self, stmt: core.Evaluate) -> None:
            if self.rejected or not isinstance(getattr(stmt.callable, "atom", None), nn.Wgmma):
                return
            lhs = stmt.args[1].type
            shard = lhs.layout
            outer = shard.layout
            row, column = outer.strides
            column = (column[0] if leading is None else leading, column[1])
            bad = replace(outer, strides=((row[0], row[1] * row_scale), column))
            operand = expr.Var("bad_lhs", type=replace(lhs, layout=replace(shard, layout=bad)))
            with pytest.raises(schedule.RefusalError, match=rf"lhs:.*{position}"):
                super().visit_Evaluate(replace(stmt, args=(stmt.args[0], operand, stmt.args[2])))
            self.rejected = True

    visitor = RejectOne()
    visitor.visit(function.body)
    assert visitor.rejected
