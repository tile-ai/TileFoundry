"""Cleanup of module context after class-body failures."""

from __future__ import annotations

from pathlib import Path

import pytest

from tilefoundry.cli.source import load_authored_ir
from tilefoundry.ir.core import VerifyError

DIAGNOSTICS = Path(__file__).parents[1] / "fixtures" / "diagnostics"


def test_failed_body_hidden_from_func() -> None:
    with pytest.raises(VerifyError, match="topology 'leaked' not declared"):
        load_authored_ir(str(DIAGNOSTICS / "dsl_leaked_topology_standalone_func.py"))


def test_failed_body_hidden_from_submodule() -> None:
    with pytest.raises(VerifyError, match="topology 'leaked' not declared"):
        load_authored_ir(str(DIAGNOSTICS / "dsl_leaked_topology_deeper_module.py"))
