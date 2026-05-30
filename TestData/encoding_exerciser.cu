// encoding_exerciser.cu — generate diverse SASS for CuAssembler encoding table quality.
// Build: nvcc -O0 -std=c++17 -arch=sm_110 -cubin -o enc_baseline.cubin encoding_exerciser.cu
// SASS:  cuobjdump --dump-sass enc_baseline.cubin > enc_baseline.sass
//
// Each __noinline__ kernel forces different register allocation, producing varied
// (register, code) pairs for the encoding matrix solver.

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>

// --- FFMA exerciser: varied register combos ---
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
__device__ __noinline__ double dfma_v1(double a, double b, double c) { return fma(a, b, c); }
__device__ __noinline__ double dfma_v2(double a, double b, double c, double d) { return fma(a, b, c) + fma(c, d, a); }

// --- FADD/FMUL exerciser ---
__device__ __noinline__ float fadd_v1(float a, float b) { return a + b; }
__device__ __noinline__ float fmul_v1(float a, float b) { return a * b; }
__device__ __noinline__ float fadd_v2(float a, float b, float c) { return a + b + c; }
__device__ __noinline__ float fmul_v2(float a, float b, float c) { return a * b * c; }

// --- Branch exerciser: force diverse BRA/BSSY/BSYNC patterns ---
__device__ __noinline__ int branch_v1(int x) {
    if (x > 0) return x * 2;
    else return x + 1;
}
__device__ __noinline__ int branch_v2(int x, int y) {
    if (x > y) return x - y;
    else if (x < y) return y - x;
    else return 0;
}
__device__ __noinline__ int branch_v3(int x) {
    int r = 0;
    for (int i = 0; i < x; i++) r += i;
    return r;
}
__device__ __noinline__ int branch_v4(int x) {
    switch (x & 3) {
        case 0: return x + 1;
        case 1: return x * 2;
        case 2: return x - 3;
        default: return x;
    }
}

// --- MUFU exerciser ---
__device__ __noinline__ float mufu_v1(float x) {
    return __frcp_rn(x) + __frsqrt_rn(x) + __sinf(x) + __cosf(x) + __expf(x) + __log2f(x);
}

// --- Integer exerciser ---
__device__ __noinline__ int imad_v1(int a, int b, int c) { return a * b + c; }
__device__ __noinline__ long long imad_wide(int a, int b, long long c) { return (long long)a * b + c; }
__device__ __noinline__ int iadd3_v1(int a, int b, int c) { return a + b + c; }
__device__ __noinline__ unsigned lop3_v1(unsigned a, unsigned b, unsigned c) {
    return (a & b) | (b ^ c) | (~a & c);
}
__device__ __noinline__ int prmt_v1(int a, int b) { return __byte_perm(a, b, 0x3210); }
__device__ __noinline__ int shf_v1(int a, int b) { return __funnelshift_l(a, b, 8); }

// --- Half-precision exerciser ---
__device__ __noinline__ __half2 hfma_v1(__half2 a, __half2 b, __half2 c) { return __hfma2(a, b, c); }
__device__ __noinline__ __half2 hadd_v1(__half2 a, __half2 b) { return __hadd2(a, b); }

// --- Memory exerciser ---
__device__ __noinline__ void ldst_v1(float* __restrict__ out, const float* __restrict__ in, int idx) {
    out[idx] = in[idx];
    out[idx + 1] = in[idx + 1];
    out[idx + 2] = in[idx + 2];
    out[idx + 3] = in[idx + 3];
}
__device__ __noinline__ void ldst_v2(int* __restrict__ out, const int* __restrict__ in, int idx) {
    out[idx] = in[idx];
    out[idx + 1] = in[idx + 1];
}

// --- Warp exerciser ---
__device__ __noinline__ int shfl_v1(int x) {
    return __shfl_xor_sync(0xffffffff, x, 1) + __shfl_down_sync(0xffffffff, x, 1);
}
__device__ __noinline__ unsigned vote_v1(int x) {
    return __ballot_sync(0xffffffff, x > 0) + __activemask();
}

// --- Atomic exerciser ---
__device__ __noinline__ void atom_v1(int* addr, int val) {
    atomicAdd(addr, val);
    atomicMax(addr, val);
    atomicMin(addr, val);
    atomicAnd(addr, val);
    atomicOr(addr, val);
    atomicXor(addr, val);
}
__device__ __noinline__ void atom_v2(float* addr, float val) {
    atomicAdd(addr, val);
}

// --- Comparison exerciser ---
__device__ __noinline__ int setp_v1(int a, int b) {
    return (a > b) + (a == b) + (a < b) + (a >= b) + (a <= b) + (a != b);
}
__device__ __noinline__ int fsetp_v1(float a, float b) {
    return (a > b) + (a == b) + (a < b);
}

// --- Select exerciser ---
__device__ __noinline__ float sel_v1(float a, float b, int c) {
    return c > 0 ? a : b;
}
__device__ __noinline__ int sel_v2(int a, int b, int c) {
    return c > 0 ? a : b;
}

// --- Special registers ---
__device__ __noinline__ unsigned sr_v1() {
    unsigned tid, ntid, ctaid, smid, warpid, laneid, clk;
    asm volatile("mov.u32 %0, %%tid.x;" : "=r"(tid));
    asm volatile("mov.u32 %0, %%ntid.x;" : "=r"(ntid));
    asm volatile("mov.u32 %0, %%ctaid.x;" : "=r"(ctaid));
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    asm volatile("mov.u32 %0, %%warpid;" : "=r"(warpid));
    asm volatile("mov.u32 %0, %%laneid;" : "=r"(laneid));
    asm volatile("mov.u32 %0, %%clock;" : "=r"(clk));
    return tid + ntid + ctaid + smid + warpid + laneid + clk;
}

// --- Conversion exerciser ---
__device__ __noinline__ float cvt_v1(int x) { return (float)x; }
__device__ __noinline__ int cvt_v2(float x) { return (int)x; }
__device__ __noinline__ double cvt_v3(float x) { return (double)x; }
__device__ __noinline__ float cvt_v4(double x) { return (float)x; }

// --- MOV exerciser ---
__device__ __noinline__ int mov_v1(int x) { return x; }
__device__ __noinline__ float mov_v2(float x) { return x; }

// --- Main kernel: call all exercisers ---
extern "C" __global__ void encoding_exerciser(float* fout, int* iout, double* dout,
                                               const float* fin, const int* iin,
                                               const __half* hin) {
    int t = threadIdx.x + blockIdx.x * blockDim.x;
    float f = fin[t];
    int i = iin[t];
    double d = (double)f;
    __half2 h2 = __halves2half2(hin[t & 31], hin[(t + 1) & 31]);

    f = ffma_v1(f, f + 1.0f, f + 2.0f);
    f += ffma_v2(f, f + 1.0f, f + 2.0f, f + 3.0f);
    f += ffma_v3(f, f + 1.0f, f + 2.0f, f + 3.0f, f + 4.0f);
    f += ffma_v4(f, f + 1.0f);
    d = dfma_v1(d, d + 1.0, d + 2.0);
    d += dfma_v2(d, d + 1.0, d + 2.0, d + 3.0);

    f += fadd_v1(f, f + 1.0f) + fmul_v1(f, f + 1.0f);
    f += fadd_v2(f, f + 1.0f, f + 2.0f) + fmul_v2(f, f + 1.0f, f + 2.0f);

    i = branch_v1(i) + branch_v2(i, i + 1) + branch_v3(i & 7) + branch_v4(i);
    f += mufu_v1(f + 1.0f);

    i += imad_v1(i, i + 1, i + 2);
    long long ll = imad_wide(i, i + 1, (long long)i);
    i += iadd3_v1(i, i + 1, i + 2);
    i += (int)lop3_v1((unsigned)i, (unsigned)(i + 1), (unsigned)(i + 2));
    i += prmt_v1(i, i + 1) + shf_v1(i, i + 1);

    h2 = hfma_v1(h2, h2, h2);
    h2 = hadd_v1(h2, h2);

    ldst_v1(fout + t * 4, fin + t * 4, 0);
    ldst_v2(iout + t * 4, iin + t * 4, 0);

    i += shfl_v1(i);
    i += (int)vote_v1(i);
    atom_v1(iout, i);
    atom_v2(fout, f);

    i += setp_v1(i, i + 1) + fsetp_v1(f, f + 1.0f);
    f = sel_v1(f, f + 1.0f, i) + (float)sel_v2(i, i + 1, i + 2);
    i += (int)sr_v1();

    f += cvt_v1(i) + cvt_v4(d);
    i += cvt_v2(f);
    d += cvt_v3(f);
    i += mov_v1(i);
    f += mov_v2(f);

    fout[t] = f;
    iout[t] = i + (int)ll;
    dout[t] = d;
}
