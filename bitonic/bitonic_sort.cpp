
#include "bitonic_sort.h"

#include "cuda.h"
#include <cstdio>
#include <iostream>

#include "../utils/utils.h"

static int THREADS_IN_BLOCK = 1024;

using namespace std;

void bitonic_sort(int* to_sort, int size) {
    cuInit(0);
    CUdevice cuDevice;
    manageResult(cuDeviceGet(&cuDevice, 0), "cannot acquire device");
    CUcontext cuContext;
    manageResult(cuCtxCreate(&cuContext, 0, cuDevice), "cannot create context");
    CUmodule cuModule = (CUmodule)0;
    manageResult(cuModuleLoad(&cuModule, "bitonic/bitonic_sort.ptx"), "cannot load module");
    CUfunction bitonic_merge;
    manageResult(cuModuleGetFunction(&bitonic_merge, cuModule, "bitonic_merge" ) , "cannot load function");
    CUfunction bitonic_triangle_merge;
    manageResult(cuModuleGetFunction(&bitonic_triangle_merge, cuModule,"bitonic_triangle_merge"), "cannot load function");

    int n;
    //fit n to power of 2
    for (n = 1; n < size; n <<= 1);
    int half_size = n/2;
    int numberOfBlocks = (half_size + THREADS_IN_BLOCK - 1) / THREADS_IN_BLOCK;
    int max_grid_dim_x = 32768;
    int x_dim = numberOfBlocks > max_grid_dim_x ? max_grid_dim_x : numberOfBlocks;
    int y_dim = (numberOfBlocks + x_dim - 1) / x_dim;

    cuMemHostRegister((void*) to_sort, size * sizeof(int), 0);
    CUdeviceptr deviceToSort;
    cuMemAlloc(&deviceToSort, size * sizeof(int));
    cuMemcpyHtoD(deviceToSort, to_sort, size * sizeof(int));


//    std::clock_t start;
//    start = std::clock();
    for (int d_traingle = 2; d_traingle <= n; d_traingle *= 2) {
        void* args1[3] = { &deviceToSort, &d_traingle, &size};

        manageResult(cuLaunchKernel(bitonic_triangle_merge, x_dim, y_dim, 1, THREADS_IN_BLOCK, 1, 1, 0, 0, args1, 0),"running");
        cuCtxSynchronize();

        for (int d = d_traingle / 4; d >= 1; d /= 2) {
            void* args2[3] = { &deviceToSort, &d, &size};
            manageResult(cuLaunchKernel(bitonic_merge, x_dim, y_dim, 1, THREADS_IN_BLOCK, 1, 1, 0, 0, args2, 0),"running");
            cuCtxSynchronize();
        }
    }
    cuCtxSynchronize();

    cuMemcpyDtoH((void*)to_sort, deviceToSort, size * sizeof(int));

    cuMemFree(deviceToSort);
    cuMemHostUnregister(to_sort);
    cuCtxDestroy(cuContext);
}

