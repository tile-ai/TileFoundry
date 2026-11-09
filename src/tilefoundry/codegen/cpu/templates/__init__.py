"""Jinja2 templates for CPU (host) codegen boilerplate.

Only the host-module wrapper is rendered here; stmt/op emission stays in the
Python walker.
"""

from __future__ import annotations

from .renderer import render
