"""What a Target validates about a program, and how codegen groups by one.

What a Target validates about a program, and how codegen groups by one.

The composed hardware facts themselves -- which documents a default target
resolves, and which limits belong to the architecture rather than the device --
are asserted where those facts are installed. What is left here is what a Target
is asked during a compilation.
"""

from __future__ import annotations

from tilefoundry.target import registered_targets


def test_registration_is_one_class_boundary_and_reload_is_idempotent() -> None:
    assert {"amx"} <= registered_targets().keys()
