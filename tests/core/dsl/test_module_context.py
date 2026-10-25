"""Cleanup of module context after class-body failures."""

from __future__ import annotations

from pathlib import Path

import pytest

from tilefoundry.cli.source import load_authored_ir
from tilefoundry.ir.core import VerifyError

DIAGNOSTICS = Path(__file__).parents[1] / "fixtures" / "diagnostics"


def test_a_raising_class_body_is_not_visible_to_a_later_standalone_func() -> None:
    with pytest.raises(VerifyError, match="topology 'leaked' not declared"):
        load_authored_ir(str(DIAGNOSTICS / "dsl_leaked_topology_standalone_func.py"))


def test_a_failed_body_is_not_inherited_by_a_deeper_module() -> None:
    with pytest.raises(VerifyError, match="topology 'leaked' not declared"):
        load_authored_ir(str(DIAGNOSTICS / "dsl_leaked_topology_deeper_module.py"))
