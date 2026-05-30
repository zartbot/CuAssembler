// variants_soup.cu — targets specific instruction variants missing from native SM_110 data.
// Focus: IADD3/IMAD/LEA multi-operand forms, FP64 with const mem, atomics with descriptors.
// Build: nvcc -O0 -std=c++17 -arch=sm_110 -cubin variants_soup.cu -o variants.cubin
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>

// --- IADD3 with predicates and .X (carry) ---
__device__ __noinline__ void iadd3_variants(long long* out, long long a, long long b, long long c) {
    // 64-bit adds generate IADD3.X with predicate outputs
    out[0] = a + b;
    out[1] = a + b + c;
    out[2] = a - b;
    out[3] = a + b + 1LL;
    out[4] = (a + b) + (c + 1LL);
    out[5] = a + b + c + 1LL;
}

// --- IMAD variants (WIDE, HI, with const, with UR) ---
__device__ __noinline__ void imad_variants(long long* out, int a, int b, int c, int d) {
    out[0] = (long long)a * b;           // IMAD.WIDE
    out[1] = (long long)a * b + c;       // IMAD.WIDE + add
    out[2] = ((long long)a * b) >> 32;   // IMAD.HI
    unsigned ua = (unsigned)a, ub = (unsigned)b;
    out[3] = (long long)ua * ub;         // IMAD.WIDE.U32
    out[4] = a * b + c;                  // IMAD
    out[5] = a * b + d;
    out[6] = a * c + d;
    out[7] = b * c + d;
}

// --- LEA variants (different shifts) ---
__device__ __noinline__ void lea_variants(int* out, int base, int idx, long long base64) {
    out[0] = base + (idx << 1);   // LEA shift=1
    out[1] = base + (idx << 2);   // LEA shift=2
    out[2] = base + (idx << 3);   // LEA shift=3
    out[3] = base + (idx << 4);   // LEA shift=4
    // 64-bit LEA
    long long r64 = base64 + ((long long)idx << 2);
    out[4] = (int)r64;
    out[5] = (int)(r64 >> 32);
    // More patterns
    out[6] = base + (idx << 5);
    out[7] = base + (idx << 6);
}

// --- DFMA/DADD/DMUL with constant memory ---
__device__ __noinline__ double dfma_const(double a, double b) {
    const double c1 = 3.14159265358979;
    const double c2 = 2.71828182845905;
    double r = fma(a, b, c1);
    r = fma(r, c2, a);
    r += c1 * a;
    r *= c2;
    r = fma(r, r, r);
    return r;
}

// --- FFMA/FADD/FMUL with constant memory ---
__device__ __noinline__ float ffma_const(float a, float b) {
    const float c1 = 3.14159f;
    const float c2 = 2.71828f;
    float r = fmaf(a, b, c1);
    r = fmaf(r, c2, a);
    r += c1 * a;
    r *= c2;
    return r;
}

// --- Global atomics with various types ---
__device__ __noinline__ void atomics_all(int* gi, unsigned* gu, float* gf,
                                          unsigned long long* gull) {
    atomicAdd(gi, 1);
    atomicSub(gi, 1);
    atomicExch(gi, 42);
    atomicMin(gi, 0);
    atomicMax(gi, 100);
    atomicAnd(gu, 0xFFFF);
    atomicOr(gu, 0xFF00);
    atomicXor(gu, 0x0F0F);
    atomicCAS(gu, 0u, 1u);
    atomicAdd(gf, 1.0f);
    atomicAdd(gull, 1ULL);
}

// --- Shared memory atomics ---
__device__ __noinline__ void atoms_all(int* si, unsigned* su) {
    atomicAdd(si, 1);
    atomicExch(si, 0);
    atomicMin(si, -100);
    atomicMax(si, 100);
    atomicCAS(su, 0u, 1u);
    atomicAnd(su, 0xFF);
    atomicOr(su, 0x100);
}

// --- LDG/STG with different vector widths and modifiers ---
__device__ __noinline__ void ldg_variants(void* out, const void* in, int t) {
    // 32-bit
    ((int*)out)[t] = ((const int*)in)[t];
    // 64-bit
    ((long long*)out)[t] = ((const long long*)in)[t];
    // 128-bit
    ((int4*)out)[t] = ((const int4*)in)[t];
}

// --- Local memory (spill) ---
__device__ __noinline__ int local_mem(int n) {
    int arr[32];
    for (int i = 0; i < 32; i++) arr[i] = i * n;
    int sum = 0;
    for (int i = 0; i < 32; i++) sum += arr[31 - i];
    return sum;
}

// --- Comparison / ISETP / FSETP with all conditions ---
__device__ __noinline__ int cmp_all(int a, int b, float fa, float fb) {
    int r = 0;
    r += (a > b) ? 1 : 0;
    r += (a >= b) ? 2 : 0;
    r += (a < b) ? 4 : 0;
    r += (a <= b) ? 8 : 0;
    r += (a == b) ? 16 : 0;
    r += (a != b) ? 32 : 0;
    r += (fa > fb) ? 64 : 0;
    r += (fa < fb) ? 128 : 0;
    r += (fa == fb) ? 256 : 0;
    return r;
}

// --- HFMA2/HADD2 with different modifiers ---
__device__ __noinline__ __half2 half_variants(__half2 a, __half2 b, __half2 c) {
    __half2 r = __hfma2(a, b, c);
    r = __hadd2(r, __hneg2(a));
    r = __hmul2(r, __habs2(b));
    r = __hfma2_sat(r, a, b);
    return r;
}

// --- SHF (funnel shift) variants ---
__device__ __noinline__ unsigned shf_all(unsigned a, unsigned b) {
    unsigned r = __funnelshift_l(a, b, 1);
    r += __funnelshift_l(a, b, 8);
    r += __funnelshift_l(a, b, 16);
    r += __funnelshift_r(a, b, 1);
    r += __funnelshift_r(a, b, 8);
    r += __funnelshift_r(a, b, 16);
    r += __funnelshift_lc(a, b, 4);
    r += __funnelshift_rc(a, b, 4);
    return r;
}

// --- PRMT (byte permute) variants ---
__device__ __noinline__ unsigned prmt_all(unsigned a, unsigned b) {
    unsigned r = __byte_perm(a, b, 0x3210);
    r += __byte_perm(a, b, 0x0123);
    r += __byte_perm(a, b, 0x7654);
    r += __byte_perm(a, b, 0x4567);
    return r;
}

// --- LOP3 variants ---
__device__ __noinline__ unsigned lop3_all(unsigned a, unsigned b, unsigned c) {
    unsigned r = (a & b) | c;
    r ^= (a | b) & c;
    r |= (~a) & b;
    r &= a ^ (b | c);
    return r;
}

// --- Main kernel ---
extern "C" __global__ void variants_soup(void* out, const void* in, int n) {
    __shared__ int si[64];
    __shared__ unsigned su[64];
    int t = threadIdx.x + blockIdx.x * blockDim.x;

    long long* lout = (long long*)out;
    int* iout = (int*)out;
    float* fout = (float*)out;
    double* dout = (double*)out;

    iadd3_variants(lout + t * 8, t, t + 1, t + 2);
    imad_variants(lout + t * 8 + 64, t, t + 1, t + 2, t + 3);
    lea_variants(iout + t * 8 + 128, t, t + 1, (long long)t);

    dout[t] = dfma_const((double)t, (double)(t + 1));
    fout[t + 256] = ffma_const((float)t, (float)(t + 1));

    atomics_all(iout, (unsigned*)iout + 1, fout + 2, (unsigned long long*)(lout + 3));
    atoms_all(si + (threadIdx.x & 63), su + (threadIdx.x & 63));
    __syncthreads();

    ldg_variants((char*)out + 4096, in, t & 15);
    iout[t + 512] = local_mem(t);
    iout[t + 1024] = cmp_all(t, n, (float)t, (float)n);

    __half2 h = __halves2half2((__half)(float)t, (__half)(float)(t+1));
    h = half_variants(h, h, h);
    fout[t + 1536] = __low2float(h);

    iout[t + 2048] = (int)shf_all((unsigned)t, (unsigned)(t + 1));
    iout[t + 2560] = (int)prmt_all((unsigned)t, (unsigned)(t + 1));
    iout[t + 3072] = (int)lop3_all((unsigned)t, (unsigned)(t + 1), (unsigned)(t + 2));
}
