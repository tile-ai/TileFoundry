"""AMX semantics translated from https://github.com/corsix/amx (MIT).

Copyright (c) 2022 Peter Cawley. See the native runtime's LICENSE.
The Python reference never issues native AMX instructions.

MIT License

Copyright (c) 2022 Peter Cawley

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""
from .state import AMXState, emulate_AMX_SET, emulate_AMX_CLR
from .memory import *
from .extract import emulate_AMX_EXTRX, emulate_AMX_EXTRY
from .genlut import emulate_AMX_GENLUT
from .mac16 import emulate_AMX_MAC16
from .matint import emulate_AMX_MATINT
from .vecint import emulate_AMX_VECINT
from .fma import *
from .fms import *
from .matfp import *
from .vecfp import *
