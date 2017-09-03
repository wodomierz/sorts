#include "cuda.h"
#include <cstdio>
#include <iostream>
#include <ctime>
#include "../utils/utils.h"

static int THREADS_IN_BLOCK = 1024;

using namespace std;


void odd_even(int *to_sort, int size) {
    cuInit(0);
    CUdevice cuDevice;
    manageResult(cuDeviceGet(&cuDevice, 0));
    CUcontext cuContext;
    manageResult(cuCtxCreate(&cuContext, 0, cuDevice));
    CUmodule cuModule = (CUmodule) 0;
    manageResult(cuModuleLoad(&cuModule, "odd-even/odd_even.ptx"));
    CUfunction odd_even_phase1;
    manageResult(cuModuleGetFunction(&odd_even_phase1, cuModule, "odd_even_phase1"));
    CUfunction odd_even_phase2;
    manageResult(cuModuleGetFunction(&odd_even_phase2, cuModule, "odd_even_phase2"));

    CUfunction odd_even;
    manageResult(cuModuleGetFunction(&odd_even, cuModule, "odd_even"));


    int n;
    int power = 0;
    //fit n to power of 2
    for (n = 1; n < size; n <<= 1, power++);
    int half_size = n / 2;

    int numberOfBlocks = (half_size + THREADS_IN_BLOCK - 1) / THREADS_IN_BLOCK;
    int max_grid_dim_x = 32768;
    int x_dim = numberOfBlocks > max_grid_dim_x ? max_grid_dim_x : numberOfBlocks;
    int y_dim = (numberOfBlocks + x_dim - 1) / x_dim;


    cuMemHostRegister((void *) to_sort, size * sizeof(int), 0);

    CUdeviceptr deviceToSort;
    cuMemAlloc(&deviceToSort, size * sizeof(int));
    cuMemcpyHtoD(deviceToSort, to_sort, size * sizeof(int));

    void *args[1] = {&deviceToSort};
    safeLaunch1Dim(odd_even, x_dim, y_dim, THREADS_IN_BLOCK, args);
    for (int pow__half_batch = 11; pow__half_batch <= power - 1; pow__half_batch++) {
        int half_batch = 1 << pow__half_batch;
        void *args1[3] = {&deviceToSort, &pow__half_batch, &size};
        safeLaunch1Dim(odd_even_phase1, x_dim, y_dim, THREADS_IN_BLOCK, args1);
        for (int d_power = pow__half_batch - 1; d_power >= 0; d_power--) {
            void *args2[4] = {&deviceToSort, &d_power, &half_batch, &size};
            safeLaunch1Dim(odd_even_phase2, x_dim, y_dim, THREADS_IN_BLOCK, args2);
        }

    }

    cuMemcpyDtoH((void *) to_sort, deviceToSort, size * sizeof(int));

    cuMemFree(deviceToSort);
    cuMemHostUnregister(to_sort);
    cuCtxDestroy(cuContext);
}

