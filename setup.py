"""Pin target extras to the version of the core distribution.

The dev extra includes libclang so runtime_spec_surface.py can read C++ headers.
"""

from pathlib import Path

from setuptools import setup
from setuptools_scm import get_version


def _version() -> str:
    """The sdist's own version, otherwise the checkout's tag.

    SETUPTOOLS_SCM_PRETEND_VERSION applies when building from a checkout.
    """
    pkg_info = Path(__file__).with_name("PKG-INFO")
    if pkg_info.is_file():
        for line in pkg_info.read_text(encoding="utf-8").splitlines():
            if line.startswith("Version: "):
                return line.removeprefix("Version: ")
    return get_version(root=".", relative_to=__file__)


version = _version()
setup(
    extras_require={
        "test": [
            "pytest>=8",
            "pytest-xdist>=3",
            "pytest-cov>=7",
        ],
        "dev": [
            "pre-commit>=3",
            "ruff>=0.14",
            "clang-format==18.1.8",
            "libclang>=18",
            "setuptools>=68",
            "setuptools-scm>=8",
        ],
        "nvidia": [f"tilefoundry-nvidia=={version}"],
        "apple": [f"tilefoundry-apple=={version}"],
        "all": ["tilefoundry[nvidia,apple]"],
    },
)
