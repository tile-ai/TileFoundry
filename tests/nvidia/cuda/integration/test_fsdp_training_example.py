"""The FSDP training example builds, checks and trains against dense PyTorch.

One participant covers the undistributed layout; two participants cover the
gather, block and backward Module slots and their sibling stages.
"""

from __future__ import annotations

import pytest

from examples.fsdp_training.model import training_program
from examples.fsdp_training.reference import check_training
from tilefoundry import analysis


@pytest.mark.parametrize("retain", [False, True], ids=["release", "retain"])
@pytest.mark.parametrize("participants", [1, 2])
def test_fsdp_training_example_matches_dense_training(participants: int, retain: bool) -> None:
    """Two momentum SGD steps match the reference model within its tolerances."""
    program = training_program(participants, retain=retain)
    module, _ = analysis.check_program(program, program.entry_function())
    errors = check_training(module)
    assert set(errors) == {"logits", "loss", "gradients", "parameters", "momentum"}
