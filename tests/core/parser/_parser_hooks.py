"""Structural parser and printer checks attached to core input cases."""

import math

import torch

from tilefoundry.dsl import *
from tilefoundry.evaluator import evaluate
from tilefoundry.ir.core import Constant


def _dtype_names(state, problems):
    def check(ir, rebuilt):
        expected = [DType.fp8e4m3, DType.f8e8m0, DType.f4e2m1]
        assert [p.type.dtype for p in ir.entry_function().params] == expected
        assert [p.type.dtype for p in rebuilt.entry_function().params] == expected

    problems.run("_dtype_names", check, state.ir, state.rebuilt)


def _carry_order(state, problems):
    def check(ir, printed, rebuilt):
        loop_lines = printed[printed.index("        for i in range(4):") :].splitlines()

        assert loop_lines[-3:] == [
            "            o = o_2",
            "            m = m_new",
            "        return (m, o)",
        ]
        assert loop_lines.index("            m_new = tf.max(m, x)") < loop_lines.index(
            "            corr = tf.sub(m, m_new)"
        )
        assert repr(ir.entry_function().body) == repr(rebuilt.entry_function().body)

    problems.run("_carry_order", check, state.ir, state.printed, state.rebuilt)


def _nested_scan(state, problems):
    def check(printed):
        lines = printed.splitlines()
        inner = next(
            i for i, line in enumerate(lines) if line.strip() == "for col in tf.tile(4, 2):"
        )
        assert lines[inner - 1].strip() != "out = out"
        assert lines[-2].strip() == "out = out"

    problems.run("_nested_scan", check, state.printed)


def _sequential_scan(state, problems):
    def check(rebuilt):
        x = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        torch.testing.assert_close(evaluate(rebuilt.entry_function(), x), (x + 1) * 2)

    problems.run("_sequential_scan", check, state.rebuilt)


def _nan_tuple(state, problems):
    def check(rebuilt):
        body = rebuilt.entry_function().body
        assert isinstance(body.elements[1], Constant)
        assert math.isnan(body.elements[1].value)

    problems.run("_nan_tuple", check, state.rebuilt)


def _nan_attr(state, problems):
    def check(rebuilt):
        assert math.isnan(rebuilt.entry_function().body.target.value)

    problems.run("_nan_attr", check, state.rebuilt)


def _scalar_range(state, problems):
    problems.expect(
        "scalar range stays a range", lambda text: "for i in tf.tile(" not in text, state.printed
    )
