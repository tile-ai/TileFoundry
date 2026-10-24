"""Pin tilefoundry to the version this package is built at."""

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
    return get_version(root="../..", relative_to=__file__)


setup(install_requires=[f"tilefoundry=={_version()}"])
