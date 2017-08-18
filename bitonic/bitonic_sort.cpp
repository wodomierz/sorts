
#include "bitonic_sort.h"

#include "cuda.h"
#include <cstdio>
#include <iostream>
#include <ctime>
#include "../utils/utils.h"

static int THREADS_IN_BLOCK = 1024;

using namespace std;

double run1(CUmodule cuModule, int power_n, CUdeviceptr deviceToSort, int size, int x_dim, int y_dim) {

    CUfunction bitonic_merge2;
    manageResult(cuModuleGetFunction(&bitonic_merge2, cuModule, "bitonic_merge2" ) , "cannot load function");

    CUfunction bitonic_merge;
    manageResult(cuModuleGetFunction(&bitonic_merge, cuModule, "bitonic_merge" ) , "cannot load function");
    CUfunction bitonic_triangle_merge;
    manageResult(cuModuleGetFunction(&bitonic_triangle_merge, cuModule,"bitonic_triangle_merge"), "cannot load function");

    std::clock_t start = std::clock();

    void* args[1] = {&deviceToSort};
    manageResult(cuLaunchKernel(bitonic_merge2, x_dim, y_dim, 1, THREADS_IN_BLOCK, 1, 1, 0, 0, args, 0),"running");
    cuCtxSynchronize();

    for (int d_half_traingle_p = 11; d_half_traingle_p <= power_n -1; d_half_traingle_p++) {
        void* args1[3] = { &deviceToSort, &d_half_traingle_p, &size};

        manageResult(cuLaunchKernel(bitonic_triangle_merge, x_dim, y_dim, 1, THREADS_IN_BLOCK, 1, 1, 0, 0, args1, 0),"running");
        cuCtxSynchronize();
        for (int d_p = d_half_traingle_p -1; d_p >= 0; d_p--) {
            void* args2[3] = { &deviceToSort, &d_p, &size};
            manageResult(cuLaunchKernel(bitonic_merge, x_dim, y_dim, 1, THREADS_IN_BLOCK, 1, 1, 0, 0, args2, 0),"running");
            cuCtxSynchronize();
        }
    }
    double delta = (std::clock() - start) / (double) (CLOCKS_PER_SEC / 1000);
    std::cout << "Time for " << "b opt wchuj" << ": " << delta << " ms"
              << std::endl;
    return delta;
}




double bitonic_sort(int* to_sort, int size, bool opt) {
    cuInit(0);
    CUdevice cuDevice;
    manageResult(cuDeviceGet(&cuDevice, 0), "cannot acquire device");
    CUcontext cuContext;
    manageResult(cuCtxCreate(&cuContext, 0, cuDevice), "cannot create context");
    CUmodule cuModule = (CUmodule)0;
    manageResult(cuModuleLoad(&cuModule, "bitonic/bitonic_sort.ptx"), "cannot load module");

    int n;
    int power_n;
    //fit n to power of 2
    for (n = 1, power_n=0; n < size; n <<= 1, power_n++);
    int half_size = n/2;
    int numberOfBlocks = (half_size + THREADS_IN_BLOCK - 1) / THREADS_IN_BLOCK;
    int max_grid_dim_x = MAX_GRID_DIM;
    int x_dim = numberOfBlocks > max_grid_dim_x ? max_grid_dim_x : numberOfBlocks;
    int y_dim = (numberOfBlocks + x_dim - 1) / x_dim;

    cuMemHostRegister((void*) to_sort, size * sizeof(int), 0);
    CUdeviceptr deviceToSort;
    cuMemAlloc(&deviceToSort, size * sizeof(int));
    cuMemcpyHtoD(deviceToSort, to_sort, size * sizeof(int));

    double result;
    result = run1(cuModule,power_n, deviceToSort, size, x_dim, y_dim);


    cuMemcpyDtoH((void*)to_sort, deviceToSort, size * sizeof(int));

    cuMemFree(deviceToSort);
    cuMemHostUnregister(to_sort);
    cuCtxDestroy(cuContext);
    return result;
}

