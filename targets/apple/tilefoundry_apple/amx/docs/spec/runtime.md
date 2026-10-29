# AMX runtime instruction reference

`include/tilefoundry/runtime/amx/instructions.h` exposes the original AMX
instruction macros from Peter Cawley's `corsix/amx` reference. The adjacent
`LICENSE` retains the MIT copyright and permission notice.

The catalogue contains24 named entries over opcodes0..22, with SET and CLR
sharing opcode17: LDX/LDY/LDZ/LDZI, STX/STY/STZ/STZI, EXTRX/EXTRY,
FMA16/32/64, FMS16/32/64, MAC16, VECFP/VECINT, MATFP/MATINT, GENLUT,
and SET/CLR. EXTRH/EXTRV are modes of the extraction opcodes.

The header is a reference for agents authoring native implementations. Arguments
are the original encoded64-bit instruction operands; SET/CLR take no GPR operand.
Callers construct the instruction's documented fields and own register state,
thread lifetime and memory. Calling a macro issues a real AMX instruction.

This package supplies no AMX code generator or automatic TIR-to-native dispatch.
Importing TIR or running its Python Evaluator does not invoke this header.
Matrix multiplication, model routers, weight packing and model contexts belong
to authored programs, rather than this instruction reference header.

The Python Evaluator translates the original C emulation semantics. Test-only
original C sources and an emulate-only wrapper provide its differential oracle;
the oracle does not include the hardware test harness or native AMX header.

The Python TIR has 39 static Op classes for these 24 instruction families.
Versioned names such as `T.amx.extrx_v2` select fixed semantics; the class's
`generation` constant configures reference evaluation, including the test-only
C simulator environment. It does not change the native opcode or expose a
public runtime version selector. The compilation Target controls which static
capabilities are admitted, and callers must choose a supported instruction.
