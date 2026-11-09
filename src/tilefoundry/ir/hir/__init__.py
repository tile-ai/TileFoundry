"""HIR Op layer: the math, nn, tensor and sharding Op categories and the scheduling carrier.

Importing the package imports every Op module, so each Op registers itself.
"""

from __future__ import annotations

from . import math, nn, schedule, sharding, tensor
