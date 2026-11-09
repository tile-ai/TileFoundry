"""HIR math Ops: element-wise binary, unary, clamp and softplus."""

from __future__ import annotations

from .binary import Binary, merge_layout
from .clamp import Clamp
from .softplus import Softplus
from .unary import Unary
