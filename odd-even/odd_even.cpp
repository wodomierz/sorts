#include "cuda.h"
#include <cstdio>
#include <iostream>
#include <cstring>
#include <algorithm>
#include <climits>

#include "odd_even.h"
#include "../utils/utils.h"

static int THREADS_IN_BLOCK = 1024;

using namespace std;

int *odd_even(int *to_sort, int size) {
    cuInit(0);
    CUdevice cuDevice;
    manageResult(cuDeviceGet(&cuDevice, 0), "cannot acquire device");
    CUcontext cuContext;
    manageResult(cuCtxCreate(&cuContext, 0, cuDevice), "cannot create Context");
    CUmodule cuModule = (CUmodule) 0;
    manageResult(cuModuleLoad(&cuModule, "odd_even.ptx"), "cannot load module");
    CUfunction odd_even_phase1;
    manageResult(cuModuleGetFunction(&odd_even_phase1, cuModule, "odd_even_phase1"), "");
    CUfunction odd_even_phase2;
    manageResult(cuModuleGetFunction(&odd_even_phase2, cuModule, "odd_even_phase2"), "");


    int numberOfBlocks = (size + THREADS_IN_BLOCK - 1) / THREADS_IN_BLOCK;
    int max_grid_dim_x = 32768;
    int x_dim = numberOfBlocks > max_grid_dim_x ? max_grid_dim_x : numberOfBlocks;
    int y_dim = (numberOfBlocks + x_dim - 1) / x_dim;


    int *result = (int *) malloc(sizeof(int) * size);
    cuMemHostRegister((void *) result, size * sizeof(int), 0);
    cuMemHostRegister((void *) to_sort, size * sizeof(int), 0);

    CUdeviceptr deviceToSort;
    cuMemAlloc(&deviceToSort, size * sizeof(int));
    cuMemcpyHtoD(deviceToSort, to_sort, size * sizeof(int));

    int n;
    //fit n to power of 2
    for (n = 1; n < size; n <<= 1);

    for (int batch_size = 1; batch_size <= n; batch_size *= 2) {
        void *args1[4] = {&deviceToSort, &batch_size, &size};
        manageResult(cuLaunchKernel(odd_even_phase1, x_dim, y_dim, 1, THREADS_IN_BLOCK, 1, 1, 0, 0, args1, 0), "running");
        for (int d = batch_size / 2; d >= 1; d *= 2) {
            void *args2[4] = {&deviceToSort, &d, &batch_size, &size};

            manageResult(cuLaunchKernel(odd_even_phase2, x_dim, y_dim, 1, THREADS_IN_BLOCK, 1, 1, 0, 0, args2, 0), "running");
            cuCtxSynchronize();
        }

    }
    cuCtxSynchronize();

    cuMemcpyDtoH((void *) result, deviceToSort, size * sizeof(int));

    cuMemFree(deviceToSort);
    cuMemHostUnregister(result);
    cuMemHostUnregister(to_sort);
    cuCtxDestroy(cuContext);
    return result;
}

