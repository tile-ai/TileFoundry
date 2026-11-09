"""Jinja2 templates for CUDA codegen boilerplate.

Only module/kernel wrappers are rendered here; stmt/op emission stays in
the Python walker (``@codegen.register(target.Target, codegen.Role.EMIT, ...)``).
"""

from __future__ import annotations

from .renderer import render
