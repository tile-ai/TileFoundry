"""The GQA decode Module of the core fixtures, declared for a CUDA target."""

from dataclasses import replace

from tests.core.fixtures.placed.gqa_decode import GqaOnline
from tilefoundry_nvidia import cuda

GqaOnline = replace(GqaOnline, target=cuda.Target("nvidia.h200_sxm"))
