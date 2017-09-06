#include "cuda.h"
#include <cstdio>
#include <iostream>
#include <ctime>
#include "../utils/utils.h"
#include "../utils/BaseData.h"

static int THREADS_IN_BLOCK = 1024;
static int BlockSize = THREADS_IN_BLOCK*2;
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


//    int n;
//    int power = 0;
    //fit n to power of 2
//    for (n = 1; n < size; n <<= 1, power++);
//    int half_size = n / 2;

    int algined_size = ceil_div(size, BlockSize) * BlockSize;
    int delta = algined_size - size;

    int n;
    int power_n;
    for (n = 1, power_n = 0; n < algined_size; n <<= 1, power_n++);

    BaseData baseData1(algined_size, BlockSize);
    BaseData baseData2(n/2, THREADS_IN_BLOCK);



//
//    int numberOfBlocks = (half_size + THREADS_IN_BLOCK - 1) / THREADS_IN_BLOCK;
//    int max_grid_dim_x = 32768;
//    int x_dim = numberOfBlocks > max_grid_dim_x ? max_grid_dim_x : numberOfBlocks;
//    int y_dim = (numberOfBlocks + x_dim - 1) / x_dim;


    cuMemHostRegister((void *) to_sort, size * sizeof(int), 0);

    CUdeviceptr deviceToSort;
    cuMemAlloc(&deviceToSort, algined_size * sizeof(int));
    CUdeviceptr shelfedToSort = addIntOffset(deviceToSort, delta);
    cuMemcpyHtoD(shelfedToSort, to_sort, size * sizeof(int));
    cuMemsetD32(deviceToSort, 0, delta);
//    cuCtxSynchronize();
//    std::clock_t start = std::clock();

    void *args[] = {&deviceToSort};
    safeLaunch1Dim(odd_even, baseData1.x_dim, baseData1.y_dim, THREADS_IN_BLOCK, args);
    for (int pow__half_batch = 11; pow__half_batch <= power_n - 1; pow__half_batch++) {
        int half_batch = 1 << pow__half_batch;
        void *args1[3] = {&deviceToSort, &pow__half_batch, &algined_size};
        safeLaunch1Dim(odd_even_phase1, baseData2.x_dim, baseData2.y_dim, THREADS_IN_BLOCK, args1);
        for (int d_power = pow__half_batch - 1; d_power >= 0; d_power--) {
            void *args2[4] = {&deviceToSort, &d_power, &half_batch, &algined_size};
            safeLaunch1Dim(odd_even_phase2,  baseData2.x_dim, baseData2.y_dim, THREADS_IN_BLOCK, args2);
        }

    }
//    cuCtxSynchronize();
//    std::clock_t end = std::clock();


    cuMemcpyDtoH((void *) to_sort, shelfedToSort, size * sizeof(int));

    cuMemFree(deviceToSort);
    cuMemHostUnregister(to_sort);
    cuCtxDestroy(cuContext);
//    return (end - start)/1000.0;
}

