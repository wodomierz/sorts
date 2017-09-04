
#include "bitonic_sort.h"

#include "cuda.h"
#include <cstdio>
#include <iostream>
#include <ctime>
#include "../utils/utils.h"

//static int ThreadsPow = 10;
static int THREADS_IN_BLOCK = 1024;
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



//    int numberOfBlocks = ceil_div(size, BlockSize);
//    int trimmed_size = numberOfBlocks * BlockSize;
//    int delta = trimmed_size - size;
    int n;
    int power_n;
    for (n = 1, power_n = 0; n < size; n <<= 1, power_n++);
    int half_size = n / 2;
    int numberOfBlocks = ceil_div(size, BlockSize);
    int x_dim = numberOfBlocks > MAX_GRID_DIM ? MAX_GRID_DIM : numberOfBlocks;
    int y_dim = (numberOfBlocks + x_dim - 1) / x_dim;

    cuMemHostRegister((void *) to_sort, size * sizeof(int), 0);
    CUdeviceptr deviceToSort;
    cuMemAlloc(&deviceToSort, size * sizeof(int));
//    CUdeviceptr shiftedDevToSort = addIntOffset(deviceToSort, delta);

    cuMemcpyHtoD(deviceToSort, to_sort, size * sizeof(int));
//    cuMemsetD32(deviceToSort, 0, delta);



    void *args[] = {&deviceToSort, &size};
    safeLaunch1Dim(bitonic_merge2, x_dim, y_dim, THREADS_IN_BLOCK, args);
    for (int d_half_traingle_p = 11; d_half_traingle_p <= power_n - 1; d_half_traingle_p++) {
        void *args1[] = {&deviceToSort, &d_half_traingle_p, &size};

        safeLaunch1Dim(bitonic_triangle_merge, x_dim, y_dim, THREADS_IN_BLOCK, args1);
        for (int d_p = d_half_traingle_p - 1; d_p >= 11; d_p--) {
            void *args2[] = {&deviceToSort, &d_p, &size};
            safeLaunch1Dim(bitonic_merge, x_dim, y_dim, THREADS_IN_BLOCK,args2);
        }
        safeLaunch1Dim(phase2_global, x_dim, y_dim, THREADS_IN_BLOCK,args);
    }

    cuMemcpyDtoH((void *) to_sort, deviceToSort, size * sizeof(int));
//    cuMemcpyDtoH((void *) to_sort, deviceToSort, size * sizeof(int));

    cuMemFree(deviceToSort);
    cuMemHostUnregister(to_sort);
    cuCtxDestroy(cuContext);
//    return (end - start);
}

