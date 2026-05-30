// mega_soup.cu — comprehensive instruction exerciser for SM_110 encoding table.
// Targets ALL instruction categories from sm_110_instruction.md.
// Build: nvcc -O0 -std=c++17 -arch=sm_110 -cubin -o mega_soup.cubin mega_soup.cu
//        nvcc -O2 -std=c++17 -arch=sm_110 -cubin -o mega_soup_O2.cubin mega_soup.cu
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cstdint>
using namespace nvcuda;

// Texture/surface (TEX/TLD/TLD4/SULD/SUST) are consumer-GPU only features:
// supported on SM_120 (RTX), NOT on SM_90/SM_100/SM_110 (datacenter).

// === FP64 intensive ===
__device__ __noinline__ double fp64_ops(double a, double b, double c) {
    double r = fma(a, b, c);        // DFMA
    r += a + b;                      // DADD
    r *= a;                          // DMUL
    r = fmin(r, a);                  // DMNMX
    r = (r > 0) ? r : -r;          // DSETP + select
    r = 1.0 / r;                    // MUFU.RCP64H
    r = sqrt(r);                    // MUFU.RSQ64H
    return r;
}

// === Integer intensive with all IADD3/IMAD/LEA variants ===
__device__ __noinline__ int int_ops_v1(int a, int b, int c, int d) {
    int r = a + b + c;              // IADD3
    r = a * b + c;                  // IMAD
    long long w = (long long)a * b; // IMAD.WIDE
    r += (int)(w >> 32);            // IMAD.HI
    r += __clz(a);                  // FLO
    r += __popc(b);                 // POPC
    r += __brev(c);                 // BREV
    r += __funnelshift_l(a, b, 8);  // SHF
    r += __byte_perm(a, b, 0x3210); // PRMT
    return r + d;
}

__device__ __noinline__ int int_ops_v2(int a, int b, int c) {
    // Force LEA variants with different shift amounts
    int r = (a << 2) + b;           // LEA
    r += (b << 3) + c;             // LEA different shift
    r += (c << 4) + a;
    r += (a << 1) + b + c;
    unsigned u = (unsigned)a;
    r += (int)__brevll((unsigned long long)u); // 64-bit ops
    r += (int)((long long)a * b + (long long)c); // IMAD.WIDE
    return r;
}

// === Predicated IADD3 / carry chain ===
__device__ __noinline__ long long int_carry(long long a, long long b) {
    // Forces IADD3.X (extended precision add)
    return a + b + 1LL;
}

// === Comparison & Selection ===
__device__ __noinline__ float cmp_ops(float a, float b, int ia, int ib) {
    float r = fminf(a, b);          // FMNMX
    r = fmaxf(r, a);                // FMNMX
    r = (a > b) ? a : b;           // FSETP + SEL
    int ir = min(ia, ib);           // IMNMX
    ir = max(ir, ia);
    ir = (ia > ib) ? ia : ib;      // ISETP + SEL
    return r + (float)ir;
}

// === Half-precision comprehensive ===
__device__ __noinline__ __half2 half_ops(__half2 a, __half2 b) {
    __half2 r = __hfma2(a, b, a);   // HFMA2
    r = __hadd2(r, b);              // HADD2
    r = __hmul2(r, a);              // HMUL2
    // min/max
    r = __hmax2(r, a);              // HMNMX2
    r = __hmin2(r, b);
    return r;
}

// === Shared memory / Local memory ===
__device__ __noinline__ void smem_ops(float* smem, float val, int idx) {
    smem[idx] = val;                // STS
    float v = smem[idx + 1];        // LDS
    smem[idx + 2] = v + 1.0f;
    // Atomics on shared
    atomicAdd(&smem[0], val);       // ATOMS
    atomicMax((int*)&smem[1], (int)val);
    atomicCAS((int*)&smem[2], 0, (int)val);
}

// === Global atomics ===
__device__ __noinline__ void atom_ops(int* gaddr, float* faddr, int val) {
    atomicAdd(gaddr, val);           // ATOMG / ATOM
    atomicSub(gaddr, val);
    atomicMax(gaddr, val);
    atomicMin(gaddr, val);
    atomicAnd(gaddr, val);
    atomicOr(gaddr, val);
    atomicXor(gaddr, val);
    atomicExch(gaddr, val);
    atomicCAS(gaddr, 0, val);
    atomicAdd(faddr, (float)val);    // ATOMG.F32
}

// === Warp shuffle/vote/reduce ===
__device__ __noinline__ int warp_ops(int val) {
    int r = __shfl_sync(0xffffffff, val, 0);        // SHFL.IDX
    r += __shfl_xor_sync(0xffffffff, val, 1);       // SHFL.BFLY
    r += __shfl_up_sync(0xffffffff, val, 1);        // SHFL.UP
    r += __shfl_down_sync(0xffffffff, val, 1);      // SHFL.DOWN
    r += __popc(__ballot_sync(0xffffffff, val > 0)); // VOTE + POPC
    r += __reduce_add_sync(0xffffffff, val);         // REDUX
    r += __reduce_min_sync(0xffffffff, (unsigned)val);
    r += __reduce_max_sync(0xffffffff, (unsigned)val);
    unsigned m = __match_any_sync(0xffffffff, val);  // MATCH
    return r + (int)m;
}

// === LDG/STG with descriptors ===
__device__ __noinline__ void ldg_stg_ops(float* out, const float* in, int t) {
    // Various load/store patterns
    float4 v4 = *((float4*)(in + t*4));   // LDG.128
    float2 v2 = *((float2*)(in + t*2));   // LDG.64
    float v1 = in[t];                      // LDG.32
    out[t] = v1 + v2.x + v4.x;
    *((float4*)(out + t*4 + 4)) = v4;     // STG.128
}

// === Conversion ops ===
__device__ __noinline__ float cvt_ops(int i, float f, double d, __half h) {
    float r = (float)i;              // I2F
    r += (float)d;                   // F2F (64->32)
    int j = (int)f;                  // F2I
    r += __int2float_rn(j);
    r += __half2float(h);            // F2F (16->32)
    __half h2 = __float2half(r);     // F2F (32->16)
    return r + __half2float(h2);
}

// === WMMA (Matrix operations) ===
__device__ __noinline__ void mma_ops(const __half* a, const __half* b, float* c) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fb;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> fc;
    wmma::fill_fragment(fc, 0.0f);
    wmma::load_matrix_sync(fa, a, 16);   // LDSM
    wmma::load_matrix_sync(fb, b, 16);
    wmma::mma_sync(fc, fa, fb, fc);      // HMMA
    wmma::store_matrix_sync(c, fc, 16, wmma::mem_row_major);  // STSM
}

// === cp.async ===
__device__ __noinline__ void cpasync_ops(float* smem, const float* gmem) {
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 16;\n\t"
        "cp.async.commit_group;\n\t"
        "cp.async.wait_group 0;\n\t"
        :: "r"((uint32_t)(uintptr_t)smem), "l"(gmem));
}

// === Control flow / barriers ===
__device__ __noinline__ int cf_ops(int x, int n) {
    int r = 0;
    // Nested branches for diverse BRA/BSSY/BSYNC patterns
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
    __syncthreads();  // BAR
    __threadfence();  // MEMBAR
    return r;
}

// === Special registers ===
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

// === Uniform path operations (UIADD3, UIMAD, ULEA, etc.) ===
__device__ __noinline__ void uniform_ops(unsigned* uout, unsigned uin) {
    // These generate U-prefixed instructions
    unsigned r = uin * 3 + 1;
    unsigned s = r << 2;
    unsigned t = (r >> 5) | (s << 3);
    uout[0] = r; uout[1] = s; uout[2] = t;
}

// === Main kernel ===
extern "C" __global__ void mega_soup(float* __restrict__ fout, int* __restrict__ iout,
                                     double* __restrict__ dout, const float* __restrict__ fin,
                                     const int* __restrict__ iin, const __half* __restrict__ hin,
                                     int n) {
    __shared__ float smem[512];
    int t = threadIdx.x + blockIdx.x * blockDim.x;

    // FP64
    double d = fp64_ops((double)t, 2.0, 1.0);

    // Integer
    int iv = int_ops_v1(t, t+1, t+2, t+3);
    iv += int_ops_v2(t, t+1, t+2);
    long long ll = int_carry((long long)t, (long long)(t+1));

    // Comparison
    float fv = cmp_ops((float)t, (float)(t+1), t, t+1);

    // FP16
    __half2 h2 = __halves2half2(hin[t & 31], hin[(t+1) & 31]);
    h2 = half_ops(h2, h2);

    // Shared mem
    smem_ops(smem, fv, threadIdx.x);
    __syncthreads();

    // Atomics
    atom_ops(iout, fout, iv);

    // Warp
    iv += warp_ops(iv);

    // LDG/STG
    ldg_stg_ops(fout + 256, fin, t & 63);

    // Conversion
    fv += cvt_ops(iv, fv, d, hin[t & 31]);

    // MMA
    if (threadIdx.x < 32) {
        mma_ops(hin, hin + 256, fout + 512);
    }

    // cp.async
    cpasync_ops(smem + threadIdx.x * 4, fin + t * 4);
    __syncthreads();

    // Control flow
    iv += cf_ops(iv, n);

    // Special registers
    iv += (int)sr_ops();

    // Uniform path
    uniform_ops((unsigned*)(iout + 256), (unsigned)t);

    // Nanosleep
    __nanosleep(8);

    // Store results
    fout[t] = fv + __low2float(h2);
    iout[t] = iv + (int)ll;
    dout[t] = d;
}
