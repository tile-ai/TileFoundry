"""Printers, viewers and reports that render IR as source, graphs and analysis text."""

from . import print_context, printer_base, tir_printer
from .analysis_data import report_data, type_text
from .analysis_report import render_analysis, render_json, render_text
from .dot import hir_function_to_dot, module_entry_to_dot
from .pattern_printer import PatternPrinter
from .print_context import HirPrintContext, PrintContext, TirPrintContext
from .printer_base import ENTRIES, ENTRY, FIELD, FIELDS, PAIR, PER_UNIT, TRIPS, PythonPrinter
from .python_printer import (
    HirPrinter,
    PythonPrintOptions,
    as_script,
    hir_function_to_python,
    module_to_python,
)
from .tir_printer import (
    TirPrinter,
    register_tir_printer,
    tir_function_to_python,
    tir_module_to_python,
)
from .values import Prose
from .viewer import Viewer
