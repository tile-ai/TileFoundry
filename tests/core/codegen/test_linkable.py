"""Target-neutral tests and helpers."""

from __future__ import annotations

from tilefoundry.codegen.linkable_module import LinkableModule


def test_a_unit_with_no_function_is_its_preamble() -> None:
    unit = LinkableModule(target="cpu", language="cpp", preamble="#include <one.h>")
    assert unit.source == "#include <one.h>\n"
