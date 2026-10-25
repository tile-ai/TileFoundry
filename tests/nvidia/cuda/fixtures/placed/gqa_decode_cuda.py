"""The GQA decode Module of the core fixtures, declared for a CUDA target."""

from dataclasses import replace

from tests.core.fixtures.placed.gqa_decode import GqaOnline
from tilefoundry_nvidia import CudaTarget

GqaOnline = replace(GqaOnline, target=CudaTarget("nvidia.h200_sxm"))
