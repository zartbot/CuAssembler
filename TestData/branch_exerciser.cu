// branch_exerciser.cu — generate diverse BRA/CALL/RET SASS for encoding table.
// The key: many __noinline__ functions at different offsets produce CALL/RET with
// varied relative address encodings. Deep branches with varied offsets produce BRA_II.
// Build: nvcc -O0 -std=c++17 -arch=sm_110 -cubin -o branch_ex.cubin branch_exerciser.cu

#include <cstdint>

// Many small __noinline__ functions to generate varied CALL/RET offsets
__device__ __noinline__ int f01(int x) { return x + 1; }
__device__ __noinline__ int f02(int x) { return x + 2; }
__device__ __noinline__ int f03(int x) { return x + 3; }
__device__ __noinline__ int f04(int x) { return x + 4; }
__device__ __noinline__ int f05(int x) { return x + 5; }
__device__ __noinline__ int f06(int x) { return x + 6; }
__device__ __noinline__ int f07(int x) { return x + 7; }
__device__ __noinline__ int f08(int x) { return x + 8; }
__device__ __noinline__ int f09(int x) { return x + 9; }
__device__ __noinline__ int f10(int x) { return x + 10; }
__device__ __noinline__ int f11(int x) { return x + 11; }
__device__ __noinline__ int f12(int x) { return x + 12; }
__device__ __noinline__ int f13(int x) { return x + 13; }
__device__ __noinline__ int f14(int x) { return x + 14; }
__device__ __noinline__ int f15(int x) { return x + 15; }
__device__ __noinline__ int f16(int x) { return x + 16; }
__device__ __noinline__ int f17(int x) { return x * 2 + 1; }
__device__ __noinline__ int f18(int x) { return x * 3 + 1; }
__device__ __noinline__ int f19(int x) { return x * 4 + 1; }
__device__ __noinline__ int f20(int x) { return x * 5 + 1; }

// Larger functions to push offsets further apart
__device__ __noinline__ int big01(int x) {
    int r = x;
    for (int i = 0; i < 10; i++) r = r * 3 + i;
    return r;
}
__device__ __noinline__ int big02(int x) {
    int r = x;
    for (int i = 0; i < 10; i++) r = r * 5 + i * 2;
    return r;
}
__device__ __noinline__ int big03(int x) {
    int r = x;
    for (int i = 0; i < 10; i++) r = r * 7 + i * 3;
    return r;
}
__device__ __noinline__ int big04(int x) {
    int r = x;
    for (int i = 0; i < 10; i++) r = r + i * i;
    return r;
}
__device__ __noinline__ int big05(int x) {
    int r = x;
    for (int i = 0; i < 10; i++) r = r ^ (r << 1) + i;
    return r;
}
__device__ __noinline__ int big06(int x) {
    int r = x;
    for (int i = 0; i < 15; i++) r = r * 11 + i;
    return r;
}
__device__ __noinline__ int big07(int x) {
    int r = x;
    for (int i = 0; i < 15; i++) r = r + i * i * i;
    return r;
}
__device__ __noinline__ int big08(int x) {
    int r = x;
    for (int i = 0; i < 20; i++) r = r * 13 + i * 7;
    return r;
}

// Branch-heavy functions for BRA_II diversity
__device__ __noinline__ int branch_deep(int x) {
    if (x > 100) {
        if (x > 200) {
            if (x > 300) return x - 300;
            else return x - 200;
        } else {
            if (x > 150) return x - 150;
            else return x - 100;
        }
    } else if (x > 50) {
        if (x > 75) return x * 2;
        else return x * 3;
    } else if (x > 25) {
        return x + 100;
    } else if (x > 10) {
        return x + 200;
    } else {
        return 0;
    }
}

__device__ __noinline__ int branch_switch(int x) {
    switch (x & 0xf) {
        case 0:  return x + 100;
        case 1:  return x + 200;
        case 2:  return x + 300;
        case 3:  return x + 400;
        case 4:  return x + 500;
        case 5:  return x + 600;
        case 6:  return x + 700;
        case 7:  return x + 800;
        case 8:  return x - 100;
        case 9:  return x - 200;
        case 10: return x - 300;
        case 11: return x - 400;
        case 12: return x - 500;
        case 13: return x - 600;
        case 14: return x - 700;
        case 15: return x - 800;
    }
    return x;
}

__device__ __noinline__ int branch_loop(int x, int n) {
    int r = 0;
    for (int i = 0; i < n; i++) {
        if (i & 1) r += x;
        else r -= x;
        if (r > 1000) break;
        if (r < -1000) break;
    }
    return r;
}

// Main kernel: calls all functions to generate CALL/RET pairs at many offsets
extern "C" __global__ void branch_exerciser(int* out, int n) {
    int t = threadIdx.x;
    int r = t;

    r = f01(r) + f02(r) + f03(r) + f04(r);
    r = f05(r) + f06(r) + f07(r) + f08(r);
    r = f09(r) + f10(r) + f11(r) + f12(r);
    r = f13(r) + f14(r) + f15(r) + f16(r);
    r = f17(r) + f18(r) + f19(r) + f20(r);

    r = big01(r) + big02(r) + big03(r) + big04(r);
    r = big05(r) + big06(r) + big07(r) + big08(r);

    r += branch_deep(r);
    r += branch_switch(r);
    r += branch_loop(r, n);

    // Additional branches from conditionals
    if (r > 0) {
        r = f01(r) + big01(r);
    } else {
        r = f10(r) + big05(r);
    }

    if (t < 16) {
        r += branch_deep(r + t);
    } else {
        r += branch_switch(r - t);
    }

    out[t] = r;
}
