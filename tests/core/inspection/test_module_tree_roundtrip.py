"""Round trips of sibling calls in a module tree."""

from __future__ import annotations

from pathlib import Path

from tests.core._corpus import reload
from tilefoundry.cli.source import load_authored_ir
from tilefoundry.inspection import as_script

SIBLINGS = Path(__file__).parents[1] / "fixtures" / "logical" / "sibling_call.py"


def test_sibling_call_survives_roundtrip() -> None:
    """A body calling an earlier sibling is canonical output.

    A body calling an earlier sibling is canonical output, so importing it
    must resolve the callee to that sibling rather than read it as an op name.
    """
    _Siblings = load_authored_ir(str(SIBLINGS))
    imported = reload(as_script(_Siblings), "_Siblings")

    assert [fn.name for fn in imported.functions] == ["callee", "caller"]
    callee, caller = imported.functions
    assert caller.body.target is callee

    once = as_script(imported)
    assert as_script(reload(once, "_Siblings")) == once
