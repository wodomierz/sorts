#include "bitonic_sort.h"

#include "cuda.h"
#include <cstdio>
#include <iostream>
#include "../utils/utils.h"
#include "../utils/BaseKernelData.h"

static const int ThreadsPow = BITONIC_THREADS_POW;
static const int BlockPow = ThreadsPow + 1;
static const int ThreadsInBlock = (1 << ThreadsPow);
static const int BlockSize = (1 << BlockPow);

using namespace std;

void bitonic_sort(int *to_sort, int size) {
    cuInit(0);
    CUdevice cuDevice;
    manageResult(cuDeviceGet(&cuDevice, 0));
    CUcontext cuContext;
    manageResult(cuCtxCreate(&cuContext, 0, cuDevice));
    CUmodule cuModule = (CUmodule) 0;
    manageResult(cuModuleLoad(&cuModule, "bitonic/bitonic_sort.ptx"));

    CUfunction bitonic_merge2;
    manageResult(cuModuleGetFunction(&bitonic_merge2, cuModule, "bitonic_merge2"));

    CUfunction phase2_global;
    manageResult(cuModuleGetFunction(&phase2_global, cuModule, "phase2_global"));

    CUfunction bitonic_merge;
    manageResult(cuModuleGetFunction(&bitonic_merge, cuModule, "bitonic_merge"));

    CUfunction bitonic_triangle_merge;
    manageResult(cuModuleGetFunction(&bitonic_triangle_merge, cuModule, "bitonic_triangle_merge"));

    int algined_size = expand_to_power_of_2(size, BlockPow);
    int delta = algined_size - size;

    int n;
    int power_n;
    for (n = 1, power_n = 0; n < algined_size; n <<= 1, power_n++);

    BaseKernelData baseData1(algined_size, BlockSize);
    BaseKernelData baseData2(n / 2, ThreadsInBlock);

    cuMemHostRegister((void *) to_sort, size * sizeof(int), 0);
    CUdeviceptr deviceToSort;

    cuMemAlloc(&deviceToSort, algined_size * sizeof(int));

    CUdeviceptr shiftedToSort = addIntOffset(deviceToSort, delta);

    cuMemcpyHtoD(shiftedToSort, to_sort, size * sizeof(int));

    cuMemsetD32(deviceToSort, 0, delta);

    void *args[] = {&deviceToSort};
    safeLaunch1Dim(bitonic_merge2, baseData1.x_dim, baseData1.y_dim, ThreadsInBlock, args);
    for (int d_half_traingle_p = ThreadsPow + 1; d_half_traingle_p <= power_n - 1; d_half_traingle_p++) {
        void *args1[] = {&deviceToSort, &d_half_traingle_p, &algined_size};

        safeLaunch1Dim(bitonic_triangle_merge, baseData2.x_dim, baseData2.y_dim, ThreadsInBlock, args1);
        for (int d_p = d_half_traingle_p - 1; d_p >= BlockPow; d_p--) {
            void *args2[] = {&deviceToSort, &d_p, &algined_size};
            safeLaunch1Dim(bitonic_merge, baseData2.x_dim, baseData2.y_dim, ThreadsInBlock, args2);
        }
        safeLaunch1Dim(phase2_global, baseData1.x_dim, baseData1.y_dim, ThreadsInBlock, args);
    }
    cuMemcpyDtoH((void *) to_sort, shiftedToSort, size * sizeof(int));

    cuMemHostUnregister(to_sort);
    cuMemFree(deviceToSort);
    cuCtxDestroy(cuContext);
}

