/* Differential wrapper only. Original MIT reference is unchanged in c/. */
#include "c/emulate.h"
#include <stddef.h>
uint32_t AMX_VER;
#define OPS(X) X(LDX) X(LDY) X(LDZ) X(LDZI) X(STX) X(STY) X(STZ) X(STZI) X(EXTRX) X(EXTRY) X(FMA16) X(FMA32) X(FMA64) X(FMS16) X(FMS32) X(FMS64) X(MAC16) X(MATFP) X(MATINT) X(VECFP) X(VECINT) X(GENLUT)
#define DECL(n) void emulate_AMX_##n(amx_state*, uint64_t);
OPS(DECL)
#define ENTRY(n) emulate_AMX_##n,
static void (*const ops[])(amx_state*,uint64_t) = { OPS(ENTRY) };
/* Only pointers and scalars cross this ABI; state alignment stays inside C. */
int amx_oracle(uint32_t op, uint32_t version, uint64_t operand,
               uint8_t* state_bytes, uint8_t* memory, size_t memory_size) {
    if (op >= sizeof(ops)/sizeof(ops[0]) || version<1 || version>4) return -1;
    if (op<8) {
        size_t offset=operand&0x00ffffffffffffffull, length=64;
        if (op<3 && (operand&(1ull<<62)))
            length=(version>=2 && (operand&(1ull<<60)) && op<2)?256:128;
        if (op>=4 && op<=6 && (operand&(1ull<<62))) length=128;
        if (offset>memory_size || length>memory_size-offset) return -2;
        uintptr_t address=(uintptr_t)(memory+offset);
        if (address>>56) return -3;
        operand=(operand&0xff00000000000000ull)|address;
    }
    amx_state state;
    memcpy(&state,state_bytes,sizeof(state));
    uint64_t old_fpcr,old_fpsr;
    __asm__ volatile("mrs %0, fpcr" : "=r"(old_fpcr));
    __asm__ volatile("mrs %0, fpsr" : "=r"(old_fpsr));
    /* RNE, DN, gradual underflow, ordinary ARM floating semantics. */
    uint64_t fpcr=(old_fpcr & ~((3ull<<22)|(1ull<<24)|(1ull<<19)|3ull))|(1ull<<25);
    __asm__ volatile("msr fpcr, %0" :: "r"(fpcr) : "memory");
    AMX_VER=version;
    ops[op](&state,operand);
    __asm__ volatile("msr fpcr, %0" :: "r"(old_fpcr) : "memory");
    __asm__ volatile("msr fpsr, %0" :: "r"(old_fpsr) : "memory");
    memcpy(state_bytes,&state,sizeof(state));
    return 0;
}
