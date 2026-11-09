"""Compiler services that lower layers call and higher layers implement."""

from .facade import (
    builtin_analyzer,
    converter,
    describe_refusal,
    dim_range,
    evaluate,
    normalize_dim,
    prepare,
    refusal_reason,
    registry,
    render,
    report_json,
    simplify_dim,
    specialize,
    verify_call,
)
