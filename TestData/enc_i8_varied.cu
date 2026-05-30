// Varied UTCIMMA kernels for encoding table diversity.
// Each kernel uses different argument order to force varied register allocation.
// Build: nvcc -O0 -std=c++17 -gencode arch=compute_110a,code=sm_110a -cubin ...
#include <cstdint>

extern "C" __global__ void i8_v1(uint32_t* o, uint64_t da, uint64_t db, uint32_t sc) {
    uint32_t tc = o[0];
    asm volatile(
        "{\n\t.reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::i8 [%0], %1, %2, %3, p;\n\t}\n"
        :: "r"(tc), "l"(da), "l"(db), "r"(0x040404A0u), "r"(sc));
    o[1] = tc;
}

extern "C" __global__ void i8_v2(uint64_t da, uint32_t* o, uint64_t db, uint32_t sc) {
    uint32_t tc = o[0];
    asm volatile(
        "{\n\t.reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::i8 [%0], %1, %2, %3, p;\n\t}\n"
        :: "r"(tc), "l"(da), "l"(db), "r"(0x040404A0u), "r"(sc));
    o[1] = tc;
}

extern "C" __global__ void i8_v3(uint64_t da, uint64_t db, uint32_t* o, uint32_t sc) {
    uint32_t tc = o[0];
    asm volatile(
        "{\n\t.reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::i8 [%0], %1, %2, %3, p;\n\t}\n"
        :: "r"(tc), "l"(da), "l"(db), "r"(0x040404A0u), "r"(sc));
    o[1] = tc;
}

extern "C" __global__ void i8_v4(uint32_t* o, uint64_t da, uint64_t db, uint32_t sc, uint32_t dummy) {
    uint32_t tc = o[0] + dummy;
    asm volatile(
        "{\n\t.reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::i8 [%0], %1, %2, %3, p;\n\t}\n"
        :: "r"(tc), "l"(da), "l"(db), "r"(0x040404A0u), "r"(sc));
    o[1] = tc;
}

extern "C" __global__ void i8_v5(uint32_t* o, uint64_t da, uint64_t db, uint32_t sc) {
    uint32_t tc = o[0];
    uint32_t tc2 = o[2];
    asm volatile(
        "{\n\t.reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::i8 [%0], %1, %2, %3, p;\n\t}\n"
        :: "r"(tc), "l"(da), "l"(db), "r"(0x040404A0u), "r"(sc));
    asm volatile(
        "{\n\t.reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::i8 [%0], %1, %2, %3, p;\n\t}\n"
        :: "r"(tc2), "l"(da), "l"(db), "r"(0x040404A0u), "r"(sc));
    o[1] = tc;
    o[3] = tc2;
}
