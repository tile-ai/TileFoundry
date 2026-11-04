"""Participant-local values retained across typed tuple calls."""

from pathlib import Path

import torch

from tilefoundry.analysis.check import check_program
from tilefoundry.cli.source import load_authored_ir
from tilefoundry.evaluator import evaluate

FIXTURE = Path(__file__).parents[1] / "fixtures" / "evaluator" / "nested_tuple_call.py"


def test_nested_tuple_argument_preserves_participant_values():
    program = load_authored_ir(f"{FIXTURE}:TupleCall")
    _, function = check_program(program, program.entry_function())
    x = torch.arange(8).float().reshape(4, 2)
    torch.testing.assert_close(evaluate(function, x), 3 * x)
