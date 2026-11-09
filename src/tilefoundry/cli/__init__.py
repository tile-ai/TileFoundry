"""Command-line interface for authored TileFoundry HIR analysis."""

from __future__ import annotations

from . import analyze, data, models, schedule, target
from .app import build_parser, main
from .source import load_authored_ir, load_namespace, one_extent_per_dim, parse_dims, select_ir
from .spec import read_spec, spec_path
