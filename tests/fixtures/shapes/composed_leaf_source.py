"""A composed root reaching one weighted child, as DSL source for the CLI to read.

The CLI tests hand a file to ``tilefoundry`` rather than importing a program, so
this shape is shared as text. ``dim_name`` is the one thing that varies: each
caller passes its own name on the command line (``--dim n_cli=4``,
``--dim n_check=4``), and one shared name would let the two files' assertions
answer for each other.
"""

from __future__ import annotations


def composed_leaf_source(dim_name: str) -> str:
    """The source a CLI test writes to a temp file, with its own DimVar name."""
    return (
        "from tilefoundry import func, module\n"
        "from tilefoundry.dsl import ConstTensor, DimVar, Tensor, tf\n"
        "from tilefoundry.target import CudaTarget\n"
        f"N = DimVar('{dim_name}', 1, 9)\n"
        "@module(entry='run')\n"
        "class Leaf:\n"
        "    @func\n"
        "    def run(x: Tensor[(N,), 'f32'], w: ConstTensor[(1,), 'f32']) -> Tensor[(N,), 'f32']:\n"
        "        return tf.mul(x, w)\n"
        "@module(entry='root', target=CudaTarget('nvidia.h200_sxm'))\n"
        "class Composed:\n"
        "    leaf = Leaf\n"
        "    @func\n"
        "    def root(x: Tensor[(N,), 'f32']) -> Tensor[(N,), 'f32']:\n"
        "        return leaf(x)\n"
    )
