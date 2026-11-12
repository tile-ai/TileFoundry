"""Code generation, including CPU host ABI registrations."""

from __future__ import annotations

from . import context, cpu, linkable_function, linkable_module, signature
from .context import EmitContext
from .context_builder import Context, build_context, mangle
from .emitter import CppEmitter
from .link_module import link_program
from .linkable_function import LinkableFunction
from .linkable_module import LinkableModule
from .options import Options
from .registry import Role, register, registry
from .scalar_expr import render_dim_call, render_scalar_expr
from .signature import (
    CallableSignature,
    LaunchSignature,
    ProgramIdSignature,
    ProgramMetaSignature,
    Signature,
    TensorSignature,
    tensor_signature_of,
)
from .visitor import Visitor
