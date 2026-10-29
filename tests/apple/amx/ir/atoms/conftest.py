"""Compile the original CPU C simulator once for the atom tests."""

import platform
import sys

import pytest

from ._c_reference import COracle, build_oracle


@pytest.fixture(scope="session")
def c_oracle(tmp_path_factory):
    # The original simulator uses ARM floating-point assembly, not AMX opcodes.
    if sys.platform != "darwin" or platform.machine().lower() != "arm64":
        pytest.skip("Original C simulator requires macOS arm64 floating-point assembly")
    return COracle(build_oracle(tmp_path_factory.mktemp("amx-original-c")))
