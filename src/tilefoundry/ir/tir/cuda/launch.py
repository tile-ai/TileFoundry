"""CUDA launch attribute selectors."""

from enum import IntEnum


class CudaLaunchAttr(IntEnum):
    """Authored selector for a CUDA launch attribute.

    The values are a subset of ``cudaLaunchAttributeID`` and are interpreted
    by CUDA target lowering when carried in ``LaunchAttrs``.
    """

    COOPERATIVE = 1
    PROGRAMMATIC_STREAM_SERIALIZATION = 2
    CLUSTER_DIMENSION = 3
