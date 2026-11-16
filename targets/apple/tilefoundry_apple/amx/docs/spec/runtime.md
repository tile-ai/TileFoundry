# AMX runtime instruction reference

`include/tilefoundry/runtime/amx/instructions.h` exposes the original AMX
instruction macros from Peter Cawley's `corsix/amx` reference. The adjacent
`LICENSE` retains the MIT copyright and permission notice.

The catalogue contains 24 named entries over opcodes 0..22. SET and CLR share
opcode 17. The entries are LDX/LDY/LDZ/LDZI, STX/STY/STZ/STZI, EXTRX/EXTRY,
FMA16/32/64, FMS16/32/64, MAC16, VECFP/VECINT, MATFP/MATINT, GENLUT,
and SET/CLR. EXTRH/EXTRV are modes of the extraction opcodes.

The header is a reference for agents that author native implementations.
Arguments are the original encoded 64-bit instruction operands. SET/CLR take no
GPR operand. Callers construct the documented fields of the instruction.
Callers own register state, thread lifetime and memory. Calling a macro issues a
real AMX instruction.

This package supplies no AMX code generator or automatic TIR-to-native dispatch.
Importing TIR or running its Python Evaluator does not invoke this header.
Authored programs own matrix multiplication, model routers, weight packing and
model contexts. The instruction reference header does not own them.

The Python Evaluator translates the original C emulation semantics. Test-only
original C sources and an emulate-only wrapper provide its differential oracle.
The oracle does not include the hardware test harness or native AMX header.

The Python TIR has 39 static Op classes for these 24 instruction families.
Versioned names such as `T.amx.extrx_v2` select fixed semantics. The class's
`generation` constant configures reference evaluation, including the test-only
C simulator environment. The constant does not change the native opcode. The
constant does not expose a public runtime version selector. The compilation
Target controls which static capabilities it admits. Callers must choose a
supported instruction.
