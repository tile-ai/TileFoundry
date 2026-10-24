"""CUDA-specific hardware markers and admission."""

import pytest


def pytest_configure(config) -> None:
    config.addinivalue_line("markers", "sm89: run only where the local CUDA device is compute capability 8.9")


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Skip an ``sm89`` test unless the local CUDA device is compute capability 8.9.

    Such a test runs a kernel on the hardware the sm_89 documents describe,
    which upstream CI does not have.
    """
    if item.get_closest_marker("sm89") is None:
        return
    import torch  # noqa: PLC0415

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 9):
        pytest.skip("needs a local compute capability 8.9 CUDA device")
