"""H14 HWX image containers and their binding descriptors."""

from . import coefficients, macho
from .bindings import TensorBinding
from .coefficients import Coefficient, CoefficientLayout
from .image import Image, coefficient_offset
from .relocations import (
    CoefficientReloc,
    Reloc,
    base_slots,
    coeff_slots,
    coefficient_relocations,
)
