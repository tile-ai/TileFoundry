"""Verification: rules registered per Op or Stmt, and the visitor that applies them."""

from .context import Context
from .layouts import require_layouts
from .registry import register, registry
from .visitor import Visitor
