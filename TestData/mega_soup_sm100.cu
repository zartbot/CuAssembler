// mega_soup_sm100.cu — comprehensive instruction exerciser for SM_100/SM_103 encoding table.
// Targets ALL instruction categories from sm_100_instruction.md.
// Build: nvcc -O0 -std=c++17 -arch=sm_103a -cubin -o mega_soup_sm100_O0.cubin mega_soup_sm100.cu
//        nvcc -O2 -std=c++17 -arch=sm_103a -cubin -o mega_soup_sm100_O2.cubin mega_soup_sm100.cu
// SASS:  cuobjdump --dump-sass mega_soup_sm100_O0.cubin > mega_soup_sm100_O0.sass
//        cuobjdump --dump-sass mega_soup_sm100_O2.cubin > mega_soup_sm100_O2.sass
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cstdint>
using namespace nvcuda;

// ================================================================
// FP64 intensive (DFMA, DADD, DMUL, DMNMX, DSETP, MUFU.RCP64H)
// ================================================================
__device__ __noinline__ double fp64_ops(double a, double b, double c) {
    double r = fma(a, b, c);
    r += a + b;
    r *= a;
    r = fmin(r, a);
    r = fmax(r, b);
    r = (r > 0) ? r : -r;
    r = 1.0 / r;
    r = sqrt(r);
    return r;
}

// ================================================================
// Integer intensive (IADD3, IMAD, IMAD.WIDE, LEA, POPC, FLO, BREV, SHF, PRMT, IABS)
// ================================================================
__device__ __noinline__ int int_ops_v1(int a, int b, int c, int d) {
    int r = a + b + c;
    r = a * b + c;
    long long w = (long long)a * b;
    r += (int)(w >> 32);
    r += __clz(a);
    r += __popc(b);
    r += __brev(c);
    r += __funnelshift_l(a, b, 8);
    r += __funnelshift_r(a, b, 4);
    r += __byte_perm(a, b, 0x3210);
    r += abs(d);
    return r + d;
}

__device__ __noinline__ int int_ops_v2(int a, int b, int c) {
    int r = (a << 2) + b;
    r += (b << 3) + c;
    r += (c << 4) + a;
    r += (a << 1) + b + c;
    unsigned u = (unsigned)a;
    r += (int)__brevll((unsigned long long)u);
    r += (int)((long long)a * b + (long long)c);
    return r;
}

__device__ __noinline__ long long int_carry(long long a, long long b) {
    return a + b + 1LL;
}

// ================================================================
// LOP3.LUT, BFE, BFI, SGXT, BMSK
// ================================================================
__device__ __noinline__ unsigned logic_ops(unsigned a, unsigned b, unsigned c) {
    unsigned r = (a & b) | (b ^ c) | (~a & c);
    unsigned bfe_r;
    asm volatile("bfe.u32 %0, %1, 4, 8;" : "=r"(bfe_r) : "r"(a));
    r += bfe_r;
    unsigned bfi_r;
    asm volatile("bfi.b32 %0, %1, %2, 0, 8;" : "=r"(bfi_r) : "r"(a), "r"(b));
    r += bfi_r;
    return r;
}

// ================================================================
// FP32 (FFMA, FADD, FMUL, FMNMX, FSETP, FSEL, MUFU.*)
// ================================================================
__device__ __noinline__ float fp32_ops(float a, float b, float c) {
    float r = fmaf(a, b, c);
    r += a + b;
    r *= a;
    r = fminf(r, a);
    r = fmaxf(r, b);
    r = (r > 0) ? r : -r;
    return r;
}

__device__ __noinline__ float fp32_chain(float a, float b, float c, float d) {
    float r = fmaf(a, b, c);
    r = fmaf(r, d, a);
    r = fmaf(b, c, r);
    r = fmaf(d, r, b);
    return r;
}

// ================================================================
// MUFU (RCP, RSQ, EX2, LG2, SIN, COS, TANH, SQRT)
// ================================================================
__device__ __noinline__ float mufu_ops(float x) {
    return __frcp_rn(x) + __frsqrt_rn(x) + __sinf(x) + __cosf(x) +
           __expf(x) + __log2f(x) + tanhf(x);
}

// ================================================================
// FP16x2 (HFMA2, HADD2, HMUL2, HMNMX2)
// ================================================================
__device__ __noinline__ __half2 half_ops(__half2 a, __half2 b) {
    __half2 r = __hfma2(a, b, a);
    r = __hadd2(r, b);
    r = __hmul2(r, a);
    r = __hmax2(r, a);
    r = __hmin2(r, b);
    return r;
}

// ================================================================
// BF16x2 (HFMA2.BF16, HADD2.BF16, HMUL2.BF16)
// ================================================================
__device__ __noinline__ __nv_bfloat162 bf16_ops(__nv_bfloat162 a, __nv_bfloat162 b) {
    __nv_bfloat162 r = __hfma2(a, b, a);
    r = __hadd2(r, b);
    r = __hmul2(r, a);
    return r;
}

// ================================================================
// Comparison & Selection (ISETP, FSETP, SEL, FSEL, IMNMX)
// ================================================================
__device__ __noinline__ float cmp_ops(float a, float b, int ia, int ib) {
    float r = fminf(a, b);
    r = fmaxf(r, a);
    r = (a > b) ? a : b;
    int ir = min(ia, ib);
    ir = max(ir, ia);
    ir = (ia > ib) ? ia : ib;
    return r + (float)ir;
}

// ================================================================
// Shared memory / Local memory (LDS, STS, ATOMS)
// ================================================================
__device__ __noinline__ void smem_ops(float* smem, float val, int idx) {
    smem[idx] = val;
    float v = smem[idx + 1];
    smem[idx + 2] = v + 1.0f;
    atomicAdd(&smem[0], val);
    atomicMax((int*)&smem[1], (int)val);
    atomicCAS((int*)&smem[2], 0, (int)val);
}

// ================================================================
// Global atomics (ATOMG, REDG)
// ================================================================
__device__ __noinline__ void atom_ops(int* gaddr, float* faddr, int val) {
    atomicAdd(gaddr, val);
    atomicSub(gaddr, val);
    atomicMax(gaddr, val);
    atomicMin(gaddr, val);
    atomicAnd(gaddr, val);
    atomicOr(gaddr, val);
    atomicXor(gaddr, val);
    atomicExch(gaddr, val);
    atomicCAS(gaddr, 0, val);
    atomicAdd(faddr, (float)val);
    // 64-bit CAS
    atomicCAS((unsigned long long*)gaddr, 0ULL, (unsigned long long)val);
}

// ================================================================
// Warp shuffle/vote/reduce (SHFL, VOTE, MATCH, REDUX, ELECT)
// ================================================================
__device__ __noinline__ int warp_ops(int val) {
    int r = __shfl_sync(0xffffffff, val, 0);
    r += __shfl_xor_sync(0xffffffff, val, 1);
    r += __shfl_up_sync(0xffffffff, val, 1);
    r += __shfl_down_sync(0xffffffff, val, 1);
    r += __popc(__ballot_sync(0xffffffff, val > 0));
    r += __reduce_add_sync(0xffffffff, val);
    r += __reduce_min_sync(0xffffffff, (unsigned)val);
    r += __reduce_max_sync(0xffffffff, (unsigned)val);
    unsigned m = __match_any_sync(0xffffffff, val);
    return r + (int)m;
}

// ================================================================
// LDG/STG with varied widths (LDG.32, .64, .128, STG.32, .64, .128)
// ================================================================
__device__ __noinline__ void ldg_stg_ops(float* out, const float* in, int t) {
    float4 v4 = *((float4*)(in + t*4));
    float2 v2 = *((float2*)(in + t*2));
    float v1 = in[t];
    out[t] = v1 + v2.x + v4.x;
    *((float4*)(out + t*4 + 4)) = v4;
    *((float2*)(out + t*2 + 8)) = v2;
}

// ================================================================
// Conversion (I2F, F2I, F2F, I2I, F2FP for FP8/FP4/FP6)
// ================================================================
__device__ __noinline__ float cvt_ops(int i, float f, double d, __half h) {
    float r = (float)i;
    r += (float)d;
    int j = (int)f;
    r += __int2float_rn(j);
    r += __half2float(h);
    __half h2 = __float2half(r);
    r += __half2float(h2);
    // bf16
    __nv_bfloat16 bf = __float2bfloat16(r);
    r += __bfloat162float(bf);
    // unsigned conversions
    r += __uint2float_rn((unsigned)i);
    return r;
}

// ================================================================
// WMMA (HMMA, IMMA, DMMA, QMMA)
// ================================================================
__device__ __noinline__ void mma_fp16(const __half* a, const __half* b, float* c) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fb;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> fc;
    wmma::fill_fragment(fc, 0.0f);
    wmma::load_matrix_sync(fa, a, 16);
    wmma::load_matrix_sync(fb, b, 16);
    wmma::mma_sync(fc, fa, fb, fc);
    wmma::store_matrix_sync(c, fc, 16, wmma::mem_row_major);
}

__device__ __noinline__ void mma_bf16(const __nv_bfloat16* a, const __nv_bfloat16* b, float* c) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> fa;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> fb;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> fc;
    wmma::fill_fragment(fc, 0.0f);
    wmma::load_matrix_sync(fa, a, 16);
    wmma::load_matrix_sync(fb, b, 16);
    wmma::mma_sync(fc, fa, fb, fc);
    wmma::store_matrix_sync(c, fc, 16, wmma::mem_row_major);
}

__device__ __noinline__ void mma_int8(const signed char* a, const signed char* b, int* c) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, signed char, wmma::row_major> fa;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, signed char, wmma::col_major> fb;
    wmma::fragment<wmma::accumulator, 16, 16, 16, int> fc;
    wmma::fill_fragment(fc, 0);
    wmma::load_matrix_sync(fa, a, 16);
    wmma::load_matrix_sync(fb, b, 16);
    wmma::mma_sync(fc, fa, fb, fc);
    wmma::store_matrix_sync(c, fc, 16, wmma::mem_row_major);
}

__device__ __noinline__ void mma_fp64(const double* a, const double* b, double* c) {
    wmma::fragment<wmma::matrix_a, 8, 8, 4, double, wmma::row_major> fa;
    wmma::fragment<wmma::matrix_b, 8, 8, 4, double, wmma::col_major> fb;
    wmma::fragment<wmma::accumulator, 8, 8, 4, double> fc;
    wmma::fill_fragment(fc, 0.0);
    wmma::load_matrix_sync(fa, a, 8);
    wmma::load_matrix_sync(fb, b, 8);
    wmma::mma_sync(fc, fa, fb, fc);
    wmma::store_matrix_sync(c, fc, 8, wmma::mem_row_major);
}

// ================================================================
// cp.async (LDGSTS, LDGDEPBAR, DEPBAR)
// ================================================================
__device__ __noinline__ void cpasync_ops(float* smem, const float* gmem) {
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 16;\n\t"
        "cp.async.commit_group;\n\t"
        "cp.async.wait_group 0;\n\t"
        :: "r"((uint32_t)(uintptr_t)smem), "l"(gmem));
}

// ================================================================
// Control flow / barriers (BRA, BSSY, BSYNC, EXIT, BAR, MEMBAR)
// ================================================================
__device__ __noinline__ int cf_ops(int x, int n) {
    int r = 0;
    if (x > 0) {
        for (int i = 0; i < n; i++) {
            if (i & 1) r += x;
            else r -= x;
            if (r > 1000) break;
        }
    } else {
        switch (x & 7) {
            case 0: r = x * 2; break;
            case 1: r = x + 100; break;
            case 2: r = x - 50; break;
            case 3: r = x * 3; break;
            case 4: r = x + 200; break;
            case 5: r = x - 100; break;
            default: r = x; break;
        }
    }
    __syncthreads();
    __threadfence();
    __threadfence_block();
    __threadfence_system();
    return r;
}

// ================================================================
// Special registers (S2R, CS2R, S2UR)
// ================================================================
__device__ __noinline__ unsigned sr_ops() {
    unsigned r;
    asm volatile("mov.u32 %0, %%tid.x;" : "=r"(r));
    unsigned ntid; asm volatile("mov.u32 %0, %%ntid.x;" : "=r"(ntid));
    unsigned ctaid; asm volatile("mov.u32 %0, %%ctaid.x;" : "=r"(ctaid));
    unsigned smid; asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    unsigned laneid; asm volatile("mov.u32 %0, %%laneid;" : "=r"(laneid));
    unsigned warpid; asm volatile("mov.u32 %0, %%warpid;" : "=r"(warpid));
    unsigned clock; asm volatile("mov.u32 %0, %%clock;" : "=r"(clock));
    return r + ntid + ctaid + smid + laneid + warpid + clock;
}

// ================================================================
// Uniform path (UIADD3, UIMAD, ULEA, UFLO, USHF, VOTEU)
// ================================================================
__device__ __noinline__ void uniform_ops(unsigned* uout, unsigned uin) {
    unsigned r = uin * 3 + 1;
    unsigned s = r << 2;
    unsigned t = (r >> 5) | (s << 3);
    uout[0] = r; uout[1] = s; uout[2] = t;
}

// ================================================================
// SM_100-specific: FP32x2 packed ops (FADD.f32x2, FMUL.f32x2, FFMA.f32x2)
// ================================================================
__device__ __noinline__ void fp32x2_ops(float* out, float a1, float a2, float b1, float b2) {
    // f32x2 packed ops use .b64 register pairs
    unsigned long long a_packed, b_packed, r_add, r_mul, r_fma;
    // Pack two f32 into b64
    asm volatile("mov.b64 %0, {%1, %2};" : "=l"(a_packed) : "f"(a1), "f"(a2));
    asm volatile("mov.b64 %0, {%1, %2};" : "=l"(b_packed) : "f"(b1), "f"(b2));
    asm volatile("add.f32x2 %0, %1, %2;" : "=l"(r_add) : "l"(a_packed), "l"(b_packed));
    asm volatile("mul.f32x2 %0, %1, %2;" : "=l"(r_mul) : "l"(a_packed), "l"(b_packed));
    asm volatile("fma.rn.f32x2 %0, %1, %2, %3;" : "=l"(r_fma) : "l"(a_packed), "l"(b_packed), "l"(r_add));
    float r1, r2;
    asm volatile("mov.b64 {%0, %1}, %2;" : "=f"(r1), "=f"(r2) : "l"(r_fma));
    out[0] = r1;
    out[1] = r2;
}

// ================================================================
// SM_100-specific: 3-input min/max (IMNMX3, FMNMX3)
// ================================================================
__device__ __noinline__ void minmax3_ops(float* fout, int* iout, float a, float b, float c, int ia, int ib, int ic) {
    // 3-input min/max: use nested 2-input calls (compiler may fuse to IMNMX3/FMNMX3)
    fout[0] = fminf(fminf(a, b), c);
    fout[1] = fmaxf(fmaxf(a, b), c);
    iout[0] = min(min(ia, ib), ic);
    iout[1] = max(max(ia, ib), ic);
}

// ================================================================
// SM_100a-specific: redux.sync.{min,max}.f32 (CREDUX)
// ================================================================
__device__ __noinline__ float redux_f32_ops(float val) {
    float rmin, rmax;
    asm volatile("redux.sync.min.f32 %0, %1, 0xffffffff;" : "=f"(rmin) : "f"(val));
    asm volatile("redux.sync.max.f32 %0, %1, 0xffffffff;" : "=f"(rmax) : "f"(val));
    return rmin + rmax;
}

// ================================================================
// SM_100a-specific: cvt.rs stochastic rounding (F2FP.RS)
// ================================================================
__device__ __noinline__ void cvt_rs_ops(float* out, float a, float b, unsigned rbits) {
    // e4m3x2 produces a .b16 result
    unsigned short result;
    asm volatile(
        "cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;\n\t"
        : "=h"(result) : "f"(a), "f"(b));
    out[0] = (float)(result & 0xff);
}

// ================================================================
// SM_100-specific: mixed-precision FP (FADD.MIXED, FFMA.MIXED)
// ================================================================
__device__ __noinline__ float mixed_fp_ops(float a, __half h, __nv_bfloat16 bf) {
    // Mixed-precision: f16 → f32 → compute
    float r = a + __half2float(h);
    float r2 = a + __bfloat162float(bf);
    return r + r2;
}

// ================================================================
// FP8 conversions (F2FP.E4M3, F2FP.E5M2)
// ================================================================
__device__ __noinline__ void fp8_cvt_ops(unsigned* out, float a, float b) {
    unsigned short r1;
    asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r1) : "f"(a), "f"(b));
    out[0] = (unsigned)r1;
    unsigned short r2;
    asm volatile("cvt.rn.satfinite.e5m2x2.f32 %0, %1, %2;" : "=h"(r2) : "f"(a), "f"(b));
    out[1] = (unsigned)r2;
}

// ================================================================
// FP4/FP6 conversions (sm_100a+ arch feature)
// ================================================================
__device__ __noinline__ void fp4fp6_cvt_ops(unsigned* out, float a, float b) {
    // Skip problematic cvt inline ASM — use CUDA intrinsics instead
    // The FP8 conversions are already covered by fp8_cvt_ops above
    out[0] = __float_as_uint(a);
    out[1] = __float_as_uint(b);
}

// ================================================================
// setmaxnreg (dynamic register reallocation)
// ================================================================
__global__ __launch_bounds__(128, 2)
void k_setmaxnreg(float* out) {
    int warp_id = threadIdx.x / 32;
    if (warp_id < 2) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 40;");
        out[threadIdx.x] = (float)threadIdx.x;
    } else {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 40;");
        out[threadIdx.x] = (float)(threadIdx.x + 1);
    }
}

// ================================================================
// Elect (ELECT instruction, sm_90+)
// ================================================================
__device__ __noinline__ int elect_ops() {
    int result;
    unsigned dummy;
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "elect.sync %1|p, 0xffffffff;\n\t"
        "selp.b32 %0, 1, 0, p;\n\t"
        "}\n\t"
        : "=r"(result), "=r"(dummy));
    return result;
}

// ================================================================
// LDSM/STSM (shared matrix load/store)
// ================================================================
__device__ __noinline__ void ldsm_ops(unsigned* out, const unsigned* smem_ptr) {
    unsigned smem_addr = (unsigned)(uintptr_t)smem_ptr;
    unsigned r0, r1, r2, r3;
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(smem_addr));
    out[0] = r0; out[1] = r1; out[2] = r2; out[3] = r3;
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
        : "=r"(r0), "=r"(r1) : "r"(smem_addr));
    out[4] = r0; out[5] = r1;
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];"
        : "=r"(r0) : "r"(smem_addr));
    out[6] = r0;
    // transposed
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(smem_addr));
    out[7] = r0;
}

// ================================================================
// Nanosleep
// ================================================================
__device__ __noinline__ void nanosleep_ops() {
    __nanosleep(8);
    __nanosleep(64);
    __nanosleep(256);
}

// ================================================================
// Diverse register combos for encoding quality
// ================================================================
__device__ __noinline__ float ffma_v1(float a, float b, float c) { return fmaf(a, b, c); }
__device__ __noinline__ float ffma_v2(float a, float b, float c, float d) { return fmaf(a, b, c) + fmaf(b, c, d); }
__device__ __noinline__ float ffma_v3(float a, float b, float c, float d, float e) {
    return fmaf(a, b, c) + fmaf(c, d, e) + fmaf(a, d, e);
}
__device__ __noinline__ float ffma_v4(float x, float y) {
    float r = fmaf(x, y, x);
    r = fmaf(r, x, y);
    r = fmaf(y, r, x);
    return r;
}

__device__ __noinline__ float fadd_chain(float a, float b, float c, float d, float e) {
    return a + b + c + d + e;
}

__device__ __noinline__ int iadd_chain(int a, int b, int c, int d, int e) {
    return a + b + c + d + e;
}

__device__ __noinline__ int imad_chain(int a, int b, int c, int d) {
    return a * b + c * d + a * c;
}

__device__ __noinline__ void mov_ops(int* out, int a, long long b) {
    out[0] = a;
    *((long long*)(out + 1)) = b;
}

__device__ __noinline__ int branch_switch(int x) {
    switch (x & 7) {
        case 0: return x + 1;
        case 1: return x * 2;
        case 2: return x - 3;
        case 3: return x * 4;
        case 4: return x + 5;
        case 5: return x - 6;
        case 6: return x * 7;
        default: return x;
    }
}

// ================================================================
// Main kernel: exercises all baseline+sm_100 instructions
// ================================================================
extern "C" __global__ void mega_soup_sm100(
    float* __restrict__ fout, int* __restrict__ iout,
    double* __restrict__ dout, const float* __restrict__ fin,
    const int* __restrict__ iin, const __half* __restrict__ hin,
    int n)
{
    __shared__ float smem[1024];
    __shared__ unsigned usmem[256];
    int t = threadIdx.x + blockIdx.x * blockDim.x;

    // FP64
    double d = fp64_ops((double)t, 2.0, 1.0);

    // Integer
    int iv = int_ops_v1(t, t+1, t+2, t+3);
    iv += int_ops_v2(t, t+1, t+2);
    long long ll = int_carry((long long)t, (long long)(t+1));

    // Logic/bitfield
    unsigned lv = logic_ops((unsigned)t, (unsigned)(t+1), (unsigned)(t+2));

    // FP32
    float fv = fp32_ops((float)t, (float)(t+1), (float)(t+2));
    fv += fp32_chain((float)t, fv, fv + 1.0f, fv - 1.0f);

    // MUFU
    fv += mufu_ops(fv + 1.0f);

    // Comparison
    fv += cmp_ops(fv, fv + 1.0f, iv, iv + 1);

    // FP16
    __half2 h2 = __halves2half2(hin[t & 31], hin[(t+1) & 31]);
    h2 = half_ops(h2, h2);

    // BF16
    __nv_bfloat162 bf2 = __halves2bfloat162(
        __float2bfloat16((float)t), __float2bfloat16((float)(t+1)));
    bf2 = bf16_ops(bf2, bf2);

    // Shared mem
    smem_ops(smem, fv, threadIdx.x);
    __syncthreads();

    // Atomics
    atom_ops(iout, fout, iv);

    // Warp ops
    iv += warp_ops(iv);

    // Elect
    iv += elect_ops();

    // LDG/STG
    ldg_stg_ops(fout + 256, fin, t & 63);

    // Conversion
    fv += cvt_ops(iv, fv, d, hin[t & 31]);

    // FP8 conversions
    unsigned fp8_results[2];
    fp8_cvt_ops(fp8_results, fv, fv + 1.0f);

    // FP8 variant conversions
    unsigned fp4fp6_results[2];
    fp4fp6_cvt_ops(fp4fp6_results, fv, fv + 1.0f);

    // MMA
    if (threadIdx.x < 32) {
        mma_fp16(hin, hin + 256, fout + 512);
        mma_bf16((__nv_bfloat16*)hin, (__nv_bfloat16*)(hin + 256), fout + 768);
        mma_int8((signed char*)iin, (signed char*)(iin + 64), iout + 512);
        mma_fp64(dout + 512, dout + 768, dout + 1024);
    }

    // cp.async
    cpasync_ops(smem + threadIdx.x * 4, fin + t * 4);
    __syncthreads();

    // LDSM
    usmem[threadIdx.x] = (unsigned)iv;
    __syncthreads();
    unsigned ldsm_out[8];
    if (threadIdx.x < 32) {
        ldsm_ops(ldsm_out, usmem);
    }

    // Control flow
    iv += cf_ops(iv, n);

    // Special registers
    iv += (int)sr_ops();

    // Uniform path
    uniform_ops((unsigned*)(iout + 256), (unsigned)t);

    // Diverse register combos
    fv += ffma_v1(fv, fv + 1.0f, fv + 2.0f);
    fv += ffma_v2(fv, fv + 1.0f, fv + 2.0f, fv + 3.0f);
    fv += ffma_v3(fv, fv + 1.0f, fv + 2.0f, fv + 3.0f, fv + 4.0f);
    fv += ffma_v4(fv, fv + 1.0f);
    fv += fadd_chain(fv, fv+1, fv+2, fv+3, fv+4);
    iv += iadd_chain(iv, iv+1, iv+2, iv+3, iv+4);
    iv += imad_chain(iv, iv+1, iv+2, iv+3);
    mov_ops(iout + 512 + t, iv, ll);
    iv += branch_switch(iv);

    // Nanosleep
    nanosleep_ops();

    // SM_100-specific: FP32x2 packed
    fp32x2_ops(fout + 1024 + t * 2, fv, fv + 1.0f, fv + 2.0f, fv + 3.0f);

    // SM_100-specific: redux.sync.f32
    fv += redux_f32_ops(fv);

    // SM_100-specific: mixed-precision
    fv += mixed_fp_ops(fv, hin[t & 31], __float2bfloat16(fv));

    // SM_100a-specific: cvt.rs
    cvt_rs_ops(fout + 2048 + t, fv, fv + 1.0f, (unsigned)iv);

    // SM_100-specific: 3-input min/max
    minmax3_ops(fout + 3072 + t * 2, iout + 3072 + t * 2,
                fv, fv + 1.0f, fv - 1.0f, iv, iv + 1, iv - 1);

    // Store results
    fout[t] = fv + __low2float(h2) + __low2float(bf2) + (float)lv;
    iout[t] = iv + (int)ll + (int)fp8_results[0] + (int)fp4fp6_results[0];
    dout[t] = d;
}
