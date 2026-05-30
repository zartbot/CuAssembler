#include <cuda.h>
#include <stdio.h>

int main() {
    CUresult r = cuInit(0);
    if (r != CUDA_SUCCESS) {
        printf("cuInit failed: %d\n", (int)r);
        return 1;
    }
    CUdevice dev;
    cuDeviceGet(&dev, 0);
    CUcontext ctx;
    cuDevicePrimaryCtxRetain(&ctx, dev);
    cuCtxSetCurrent(ctx);

    CUmodule mod;
    r = cuModuleLoad(&mod, "/tmp/tcgen05_sm_110a.reasm.cubin");
    if (r != CUDA_SUCCESS) {
        printf("cuModuleLoad FAILED: %d\n", (int)r);
        return 1;
    }
    printf("cuModuleLoad OK\n");

    CUfunction f1, f2;
    r = cuModuleGetFunction(&f1, mod, "probe_mma_f16");
    printf("probe_mma_f16: %s\n", r == CUDA_SUCCESS ? "found" : "NOT FOUND");
    r = cuModuleGetFunction(&f2, mod, "probe_tmem");
    printf("probe_tmem: %s\n", r == CUDA_SUCCESS ? "found" : "NOT FOUND");

    printf("SUCCESS: reassembled cubin loads OK.\n");
    cuModuleUnload(mod);
    cuDevicePrimaryCtxRelease(dev);
    return 0;
}
