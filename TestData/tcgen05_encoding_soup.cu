// Comprehensive tcgen05/TMEM instruction soup for SM_110 encoding table generation.
// Compile with: nvcc -O0 -arch=sm_110a -cubin -o tcgen05_encoding_soup.cubin tcgen05_encoding_soup.cu
//
// Each kernel uses different UR register assignments to provide varied encoding data
// for the CuInsAssemblerRepos matrix solver.

#include <cuda.h>
#include <cstdint>

// ------ Kernel 1: UTCHMMA with varied UR registers ------
extern "C" __global__ void k_utchmma_v1(uint64_t *out, uint64_t desc_a, uint64_t desc_b,
                                         uint64_t idesc_val, uint64_t nCols) {
    asm volatile(
        "{\n\t"
        ".reg .pred p0;\n\t"
        ".reg .b32 tmem_addr, tmem_addr2;\n\t"
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 tmem_addr, %4;\n\t"
        "tcgen05.mma.cta_group::1.kind::f16 [tmem_addr], %1, %2, %3, p0;\n\t"
        "tcgen05.dealloc.cta_group::1.sync.aligned.b32 tmem_addr, %4;\n\t"
        "}\n\t"
        :
        : "l"(out), "l"(desc_a), "l"(desc_b), "r"((uint32_t)idesc_val), "r"((uint32_t)nCols)
        : "memory"
    );
}

// ------ Kernel 2: UTCHMMA.2CTA (dual-SM) ------
extern "C" __global__ void k_utchmma_2cta(uint64_t *out, uint64_t desc_a, uint64_t desc_b,
                                           uint64_t idesc_val, uint64_t nCols) {
    asm volatile(
        "{\n\t"
        ".reg .pred p0;\n\t"
        ".reg .b32 tmem_addr;\n\t"
        "tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 tmem_addr, %4;\n\t"
        "tcgen05.mma.cta_group::2.kind::f16 [tmem_addr], %1, %2, %3, p0;\n\t"
        "tcgen05.dealloc.cta_group::2.sync.aligned.b32 tmem_addr, %4;\n\t"
        "}\n\t"
        :
        : "l"(out), "l"(desc_a), "l"(desc_b), "r"((uint32_t)idesc_val), "r"((uint32_t)nCols)
        : "memory"
    );
}

// ------ Kernel 3: UTCHMMA bf16 ------
extern "C" __global__ void k_utchmma_bf16(uint64_t *out, uint64_t desc_a, uint64_t desc_b,
                                           uint64_t idesc_val, uint64_t nCols) {
    asm volatile(
        "{\n\t"
        ".reg .pred p0;\n\t"
        ".reg .b32 tmem_addr;\n\t"
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 tmem_addr, %4;\n\t"
        "tcgen05.mma.cta_group::1.kind::bf16 [tmem_addr], %1, %2, %3, p0;\n\t"
        "tcgen05.dealloc.cta_group::1.sync.aligned.b32 tmem_addr, %4;\n\t"
        "}\n\t"
        :
        : "l"(out), "l"(desc_a), "l"(desc_b), "r"((uint32_t)idesc_val), "r"((uint32_t)nCols)
        : "memory"
    );
}

// ------ Kernel 4: UTCQMMA dense (f8f6f4) ------
extern "C" __global__ void k_utcqmma_dense(uint64_t *out, uint64_t desc_a, uint64_t desc_b,
                                             uint64_t idesc_val, uint64_t nCols) {
    asm volatile(
        "{\n\t"
        ".reg .pred p0;\n\t"
        ".reg .b32 tmem_addr;\n\t"
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 tmem_addr, %4;\n\t"
        "tcgen05.mma.cta_group::1.kind::f8f6f4 [tmem_addr], %1, %2, %3, p0;\n\t"
        "tcgen05.dealloc.cta_group::1.sync.aligned.b32 tmem_addr, %4;\n\t"
        "}\n\t"
        :
        : "l"(out), "l"(desc_a), "l"(desc_b), "r"((uint32_t)idesc_val), "r"((uint32_t)nCols)
        : "memory"
    );
}

// ------ Kernel 5: UTCQMMA block-scaled (mxf8f6f4) ------
extern "C" __global__ void k_utcqmma_scaled(uint64_t *out, uint64_t desc_a, uint64_t desc_b,
                                              uint64_t idesc_val, uint64_t nCols, uint32_t sf_addr) {
    asm volatile(
        "{\n\t"
        ".reg .pred p0;\n\t"
        ".reg .b32 tmem_addr, tmem_sf;\n\t"
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 tmem_addr, %4;\n\t"
        "mov.b32 tmem_sf, %5;\n\t"
        "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale [tmem_addr], %1, %2, %3, tmem_sf, p0;\n\t"
        "tcgen05.dealloc.cta_group::1.sync.aligned.b32 tmem_addr, %4;\n\t"
        "}\n\t"
        :
        : "l"(out), "l"(desc_a), "l"(desc_b), "r"((uint32_t)idesc_val), "r"((uint32_t)nCols), "r"(sf_addr)
        : "memory"
    );
}

// ------ Kernel 6: UTCOMMA (mxf4 block-scaled) ------
extern "C" __global__ void k_utcomma(uint64_t *out, uint64_t desc_a, uint64_t desc_b,
                                      uint64_t idesc_val, uint64_t nCols, uint32_t sf_addr) {
    asm volatile(
        "{\n\t"
        ".reg .pred p0;\n\t"
        ".reg .b32 tmem_addr, tmem_sf;\n\t"
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 tmem_addr, %4;\n\t"
        "mov.b32 tmem_sf, %5;\n\t"
        "tcgen05.mma.cta_group::1.kind::mxf4.block_scale [tmem_addr], %1, %2, %3, tmem_sf, p0;\n\t"
        "tcgen05.dealloc.cta_group::1.sync.aligned.b32 tmem_addr, %4;\n\t"
        "}\n\t"
        :
        : "l"(out), "l"(desc_a), "l"(desc_b), "r"((uint32_t)idesc_val), "r"((uint32_t)nCols), "r"(sf_addr)
        : "memory"
    );
}

// ------ Kernel 7: UTCIMMA (i8, arch-only) ------
extern "C" __global__ void k_utcimma(uint64_t *out, uint64_t desc_a, uint64_t desc_b,
                                      uint64_t idesc_val, uint64_t nCols) {
    asm volatile(
        "{\n\t"
        ".reg .pred p0;\n\t"
        ".reg .b32 tmem_addr;\n\t"
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 tmem_addr, %4;\n\t"
        "tcgen05.mma.cta_group::1.kind::i8 [tmem_addr], %1, %2, %3, p0;\n\t"
        "tcgen05.dealloc.cta_group::1.sync.aligned.b32 tmem_addr, %4;\n\t"
        "}\n\t"
        :
        : "l"(out), "l"(desc_a), "l"(desc_b), "r"((uint32_t)idesc_val), "r"((uint32_t)nCols)
        : "memory"
    );
}

// ------ Kernel 8: LDTM/STTM with varied shapes ------
extern "C" __global__ void k_tmem_shapes(uint64_t *out, uint64_t nCols) {
    asm volatile(
        "{\n\t"
        ".reg .pred p0;\n\t"
        ".reg .b32 tmem_addr, r0, r1, r2, r3;\n\t"
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 tmem_addr, %1;\n\t"
        // st then ld - 32x32b shape
        "mov.b32 r0, 0;\n\t"
        "tcgen05.st.sync.aligned.32x32b.x1.b32 [tmem_addr], {r0};\n\t"
        "tcgen05.ld.sync.aligned.32x32b.x1.b32 {r1}, [tmem_addr];\n\t"
        // 16x64b shape
        "tcgen05.st.sync.aligned.16x64b.x1.b32 [tmem_addr], {r0};\n\t"
        "tcgen05.ld.sync.aligned.16x64b.x1.b32 {r2}, [tmem_addr];\n\t"
        // 16x128b shape
        "tcgen05.st.sync.aligned.16x128b.x1.b32 [tmem_addr], {r0};\n\t"
        "tcgen05.ld.sync.aligned.16x128b.x1.b32 {r3}, [tmem_addr];\n\t"
        "tcgen05.dealloc.cta_group::1.sync.aligned.b32 tmem_addr, %1;\n\t"
        "st.global.b32 [%0], r1;\n\t"
        "st.global.b32 [%0+4], r2;\n\t"
        "st.global.b32 [%0+8], r3;\n\t"
        "}\n\t"
        :
        : "l"(out), "r"((uint32_t)nCols)
        : "memory"
    );
}

// ------ Kernel 9: LDTM/STTM with multi-register (.x2, .x4) ------
extern "C" __global__ void k_tmem_multi(uint64_t *out, uint64_t nCols) {
    asm volatile(
        "{\n\t"
        ".reg .pred p0;\n\t"
        ".reg .b32 tmem_addr, r0, r1, r2, r3, r4, r5, r6, r7;\n\t"
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 tmem_addr, %1;\n\t"
        "mov.b32 r0, 0;\n\t"
        "mov.b32 r1, 1;\n\t"
        "mov.b32 r2, 2;\n\t"
        "mov.b32 r3, 3;\n\t"
        // .x2
        "tcgen05.st.sync.aligned.32x32b.x2.b32 [tmem_addr], {r0, r1};\n\t"
        "tcgen05.ld.sync.aligned.32x32b.x2.b32 {r4, r5}, [tmem_addr];\n\t"
        // .x4
        "tcgen05.st.sync.aligned.32x32b.x4.b32 [tmem_addr], {r0, r1, r2, r3};\n\t"
        "tcgen05.ld.sync.aligned.32x32b.x4.b32 {r4, r5, r6, r7}, [tmem_addr];\n\t"
        "tcgen05.dealloc.cta_group::1.sync.aligned.b32 tmem_addr, %1;\n\t"
        "st.global.b32 [%0], r4;\n\t"
        "}\n\t"
        :
        : "l"(out), "r"((uint32_t)nCols)
        : "memory"
    );
}

// ------ Kernel 10: UTCATOMSWS variants ------
extern "C" __global__ void k_utcatomsws(uint64_t *out, uint64_t nCols) {
    asm volatile(
        "{\n\t"
        ".reg .pred p0, p1;\n\t"
        ".reg .b32 tmem_addr;\n\t"
        // alloc
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 tmem_addr, %1;\n\t"
        // relinquish
        "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;\n\t"
        // dealloc
        "tcgen05.dealloc.cta_group::1.sync.aligned.b32 tmem_addr, %1;\n\t"
        "}\n\t"
        :
        : "l"(out), "r"((uint32_t)nCols)
        : "memory"
    );
}

// ------ Kernel 11: tcgen05 fence/commit/wait ------
extern "C" __global__ void k_tcgen05_sync(uint64_t *out, uint64_t desc_a, uint64_t desc_b,
                                           uint64_t idesc_val, uint64_t nCols) {
    __shared__ uint64_t mbar;
    asm volatile(
        "{\n\t"
        ".reg .pred p0;\n\t"
        ".reg .b32 tmem_addr, r0;\n\t"
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 tmem_addr, %4;\n\t"
        // fence before
        "tcgen05.fence::before_thread_sync;\n\t"
        // mma
        "tcgen05.mma.cta_group::1.kind::f16 [tmem_addr], %1, %2, %3, p0;\n\t"
        // commit
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cta.b64 [%5];\n\t"
        // fence after
        "tcgen05.fence::after_thread_sync;\n\t"
        // ld
        "tcgen05.ld.sync.aligned.32x32b.x1.b32 {r0}, [tmem_addr];\n\t"
        // wait ld
        "tcgen05.wait::ld.sync.aligned;\n\t"
        "tcgen05.dealloc.cta_group::1.sync.aligned.b32 tmem_addr, %4;\n\t"
        "st.global.b32 [%0], r0;\n\t"
        "}\n\t"
        :
        : "l"(out), "l"(desc_a), "l"(desc_b), "r"((uint32_t)idesc_val),
          "r"((uint32_t)nCols), "l"(&mbar)
        : "memory"
    );
}

// ------ Kernel 12: UTCHMMA with different register combos (for encoding diversity) ------
extern "C" __global__ void k_utchmma_v2(uint64_t *out, uint64_t desc_a, uint64_t desc_b,
                                         uint32_t idesc_val, uint32_t nCols,
                                         uint64_t desc_a2, uint64_t desc_b2, uint32_t idesc_val2) {
    asm volatile(
        "{\n\t"
        ".reg .pred p0;\n\t"
        ".reg .b32 tmem_addr;\n\t"
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 tmem_addr, %4;\n\t"
        "tcgen05.mma.cta_group::1.kind::f16 [tmem_addr], %1, %2, %3, p0;\n\t"
        "tcgen05.mma.cta_group::1.kind::f16 [tmem_addr], %5, %6, %7, p0;\n\t"
        "tcgen05.dealloc.cta_group::1.sync.aligned.b32 tmem_addr, %4;\n\t"
        "}\n\t"
        :
        : "l"(out), "l"(desc_a), "l"(desc_b), "r"(idesc_val), "r"(nCols),
          "l"(desc_a2), "l"(desc_b2), "r"(idesc_val2)
        : "memory"
    );
}
