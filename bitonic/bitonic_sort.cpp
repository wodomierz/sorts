
#include "bitonic_sort.h"

#include "cuda.h"
#include <cstdio>
#include <iostream>
#include <ctime>
#include "../utils/utils.h"
#include "../utils/BaseData.h"

static int ThreadsPow = 10;
static int THREADS_IN_BLOCK = (1 << ThreadsPow);
static int BlockSize = THREADS_IN_BLOCK *2;
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


    int algined_size = ceil_div(size, BlockSize) * BlockSize;
    int delta = algined_size - size;

    int n;
    int power_n;
    for (n = 1, power_n = 0; n < algined_size; n <<= 1, power_n++);

    BaseData baseData1(algined_size, BlockSize);
    BaseData baseData2(n/2, THREADS_IN_BLOCK);

    cuMemHostRegister((void *) to_sort, size * sizeof(int), 0);
    CUdeviceptr deviceToSort;

    cuMemAlloc(&deviceToSort, algined_size * sizeof(int));

    CUdeviceptr shiftedToSort = addIntOffset(deviceToSort, delta);

    cuMemcpyHtoD(shiftedToSort, to_sort, size * sizeof(int));
    cuMemsetD32(deviceToSort, 0, delta);

    void *args[] = {&deviceToSort};
    safeLaunch1Dim(bitonic_merge2, baseData1.x_dim, baseData1.y_dim, THREADS_IN_BLOCK, args);
    for (int d_half_traingle_p = ThreadsPow + 1; d_half_traingle_p <= power_n - 1; d_half_traingle_p++) {
        void *args1[] = {&deviceToSort, &d_half_traingle_p, &algined_size};

        safeLaunch1Dim(bitonic_triangle_merge, baseData2.x_dim, baseData2.y_dim, THREADS_IN_BLOCK, args1);
        for (int d_p = d_half_traingle_p - 1; d_p >= ThreadsPow + 1; d_p--) {
            void *args2[] = {&deviceToSort, &d_p, &algined_size};
            safeLaunch1Dim(bitonic_merge,  baseData2.x_dim, baseData2.y_dim, THREADS_IN_BLOCK,args2);
        }
        safeLaunch1Dim(phase2_global, baseData1.x_dim, baseData1.y_dim, THREADS_IN_BLOCK,args);
    }

    cuMemcpyDtoH((void *) to_sort, shiftedToSort, size * sizeof(int));

    cuMemFree(deviceToSort);
    cuMemHostUnregister(to_sort);
    cuCtxDestroy(cuContext);
}

