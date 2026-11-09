"""Participant-local values retained across typed tuple calls."""

from pathlib import Path

import torch

from tilefoundry import analysis, cli
from tilefoundry.interfaces import evaluator

FIXTURE = Path(__file__).parents[1] / "fixtures" / "evaluator" / "nested_tuple_call.py"


def test_nested_tuple_argument_preserves_participant_values():
    program = cli.load_authored_ir(f"{FIXTURE}:TupleCall")
    _, function = analysis.check_program(program, program.entry_function())
    x = torch.arange(8).float().reshape(4, 2)
    torch.testing.assert_close(evaluator.evaluate(function, x), 3 * x)
